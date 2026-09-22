//! Typed source custody and native role loan for the shared capture protocol.
use super::*;

pub(crate) trait CaptureSourceCustody:
    Clone + std::fmt::Debug + Send + Sync + 'static
{
    fn source(&self) -> &RetainedCommunicationSource;
    fn funding(&self) -> &HostMetadataFunding;
}
impl CaptureSourceCustody for Custody {
    fn source(&self) -> &RetainedCommunicationSource {
        &self.source
    }
    fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
}

/// The protocol worker sees an exact source cursor and a genuine native owner.
/// This interface cannot turn source descriptions into a model or text grant.
pub(crate) trait CaptureSourceOwner: Sized + 'static {
    type Custody: CaptureSourceCustody;
    fn retained(&self) -> Self;
    fn request(&self) -> &OriginalParallelControlRequest;
    fn custody(&self) -> &Self::Custody;
    fn running(&self) -> &Cell<bool>;
    fn failed(&self) -> &Cell<bool>;
    fn validate_active(&self) -> Result<(), Error>;
    fn communication_source(&self) -> Result<OriginalCommunicationSource<'_>, Error> {
        self.request()
            .source
            .communication_source_funded(self.custody().funding())
    }
    fn prepare_agreement(
        &self,
        event: ParallelControlEvent,
        target: Option<&Group>,
    ) -> Result<OriginalParallelControlInvocation, Error> {
        self.request()
            .prepare_with_funding(event, target, self.custody().funding())
    }
    fn run_native<I: 'static, T, E, F>(
        &self,
        invocation: I,
        capacity: AgreementCapacity,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>;
    fn agreement_context_size<T, E, F>() -> usize
    where
        F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
    {
        agreement_context_size::<T, E, F>()
    }
    fn run_agreement<T, E, F>(
        &self,
        invocation: OriginalParallelControlInvocation,
        stream: &Stream,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
    {
        let capacity = invocation.capacity();
        self.run_native(invocation, capacity, agreement_context(stream, run))
    }
}
impl CaptureSourceOwner for OriginalParallelControlOwner {
    type Custody = Custody;
    fn retained(&self) -> Self {
        Self(self.0.clone())
    }
    fn request(&self) -> &OriginalParallelControlRequest {
        &self.owner().request
    }
    fn custody(&self) -> &Custody {
        &self.owner().custody
    }
    fn running(&self) -> &Cell<bool> {
        &self.owner().running
    }
    fn failed(&self) -> &Cell<bool> {
        &self.owner().failed
    }
    fn validate_active(&self) -> Result<(), Error> {
        if self.failed().get() {
            Err(fail(CaptureCause::Identity, self.custody()))
        } else {
            Ok(())
        }
    }
    fn run_native<I: 'static, T, E, F>(
        &self,
        invocation: I,
        capacity: AgreementCapacity,
        run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where
        F: FnOnce(&I, &OriginalScopeObserver) -> Result<Result<T, E>, Error>,
    {
        run_native_role(
            invocation,
            capacity,
            &self.owner().native,
            &self.owner().custody,
            run,
        )
    }
}

fn agreement_context<'a, T, E, F>(
    stream: &'a Stream,
    run: F,
) -> impl FnOnce(
    &OriginalParallelControlInvocation,
    &OriginalScopeObserver,
) -> Result<Result<T, E>, Error>
+ 'a
where
    F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E> + 'a,
{
    move |invocation, observer| invocation.with_context(observer, stream, run)
}
pub(super) fn agreement_context_size<T, E, F>() -> usize
where
    F: FnOnce(&Group, &HostMetadataFunding) -> Result<T, E>,
{
    fn result_size<A, B, R>(_: impl FnOnce(A, B) -> R) -> usize {
        size_of::<R>()
    }
    result_size(agreement_context::<T, E, F>)
}
