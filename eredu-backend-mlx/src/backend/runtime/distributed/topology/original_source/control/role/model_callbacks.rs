//! Actual callback factories shared by execution and cold frame queries.
use super::*;
use crate::backend::submission_recovery::native_role::{self, NativeRoleContext};

pub(super) fn group_callback<'a, T, E, F>(
    projection: &'a OriginalParallelControlProjection,
    event: ParallelControlEvent,
    group: &'a Group,
    funding: &'a HostMetadataFunding,
    executor: &'a Stream,
    run: F,
) -> impl FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<Result<T, E>, Error> + use<'a, T, E, F>
where
    F: FnOnce(Option<&Group>) -> Result<T, E>,
{
    move |prepared| {
        let Some((prepared, actual_funding)) = prepared else {
            return Err(Error::with_original_control_source(
                projection.fallback.retained().into_failure(),
                false,
            ));
        };
        if !funding.same_account(actual_funding) {
            return Err(control_error(
                ControlCause::Identity,
                &projection.custody.source,
                funding,
            ));
        }
        let binding = prepared.original_control().ok_or_else(|| {
            Error::with_original_control_source(
                projection.fallback.retained().into_failure(),
                false,
            )
        })?;
        binding.with_group(event, group, prepared, actual_funding, executor, run)
    }
}
pub(super) fn option_callback<T, E, F>(
    run: F,
) -> impl FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>
where
    F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
{
    move |group, funding| run(Some((group, funding)))
}
pub(super) fn native_callback<'a, T, E, F>(
    stream: &'a Stream,
    run: F,
) -> impl FnOnce(
    &OriginalParallelControlInvocation,
    &OriginalScopeObserver,
) -> Result<Result<T, E>, Error>
+ use<'a, T, E, F>
where
    F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
{
    move |invocation, observer| invocation.with_context(observer, stream, run)
}
pub(super) fn observer_callback<I, T, E, F>(
    run: F,
) -> impl FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>
where
    F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
{
    move |invocation, context| run(invocation, context.observer())
}

fn observer_controls<I, T, E, F>() -> Option<usize>
where
    F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
{
    fn describe<I, T, E, F, N>(_: impl FnOnce(F) -> N) -> Option<usize>
    where
        N: FnOnce(&I, &NativeRoleContext<'_>) -> Result<Result<T, E>, Error>,
    {
        native_role::callback_control_bytes::<T, E>(size_of::<N>())
    }
    describe(observer_callback::<I, T, E, F>)
}
fn native_controls<T, E, F>() -> Option<usize>
where
    F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
{
    fn describe<'a, T, E, F, N>(_: impl FnOnce(&'a Stream, F) -> N) -> Option<usize>
    where
        N: FnOnce(
            &OriginalParallelControlInvocation,
            &OriginalScopeObserver,
        ) -> Result<Result<T, E>, Error>,
    {
        metadata_bytes(&[
            size_of::<OriginalParallelControlInvocation>(),
            size_of::<N>(),
            size_of::<AgreementCapacity>(),
            size_of::<(&NativeOwner, &Custody)>(),
            size_of::<Result<Result<T, E>, eredu_core::BackendFailure>>(),
        ])?
        .checked_add(observer_controls::<
            OriginalParallelControlInvocation,
            T,
            E,
            N,
        >()?)?
        .checked_add(OriginalParallelControlInvocation::context_control_bytes::<
            T,
            E,
        >(size_of::<F>())?)
    }
    describe(native_callback::<T, E, F>)
}
fn option_controls<T, E, F>() -> Option<usize>
where
    F: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<T, E>,
{
    fn describe<T, E, F, N>(_: impl FnOnce(F) -> N) -> Option<usize>
    where
        N: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
    {
        native_controls::<T, E, N>()
    }
    describe(option_callback::<T, E, F>)
}
pub(super) fn group_controls<T, E, F>() -> Option<usize>
where
    F: FnOnce(Option<&Group>) -> Result<T, E>,
{
    fn describe<'a, T, E, F, N>(
        _: impl FnOnce(
            &'a OriginalParallelControlProjection,
            ParallelControlEvent,
            &'a Group,
            &'a HostMetadataFunding,
            &'a Stream,
            F,
        ) -> N,
    ) -> Option<usize>
    where
        N: FnOnce(Option<(&Group, &HostMetadataFunding)>) -> Result<Result<T, E>, Error>,
    {
        option_controls::<Result<T, E>, Error, N>()
    }
    describe(group_callback::<T, E, F>)
}
