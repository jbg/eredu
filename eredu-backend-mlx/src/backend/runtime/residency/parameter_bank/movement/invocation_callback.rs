//! Fixed borrowed invocation worker. The caller retains its typed failure.
use super::*;
use eredu_nn::{TensorParallelGroupedOutput, workspace::HostMetadataFunding};
use eredu_runtime::expert::{
    IndexedDemandLoanError, IndexedInvocationCallback, IndexedInvocationRequest,
};
use std::mem::{size_of, size_of_val};

type Output = TensorParallelGroupedOutput<MlxTensor>;
type Callback<'a> = dyn IndexedInvocationCallback<MlxTensor, MlxIndexedMovement> + 'a;
type Failure = IndexedDemandLoanError<Error>;

#[derive(Debug, thiserror::Error)]
#[error("addressable architecture callback aborted")]
struct CallbackAborted;
fn aborted() -> Error {
    Error::Other(Box::new(CallbackAborted))
}
fn overflow() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
}
fn identity() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}

/// Only a borrowed, object-safe owner enters the native invocation guard.
/// Architecture provider and error layouts never enter this worker's census.
struct Owner<'a> {
    callback: &'a mut Callback<'a>,
    funding: &'a HostMetadataFunding,
}
impl Owner<'_> {
    fn movement(owner: &mut Self) -> &mut MlxIndexedMovement {
        owner.callback.movement()
    }
    fn body(owner: &mut Self) -> Result<Output, ()> {
        owner.callback.run(Some(owner.funding))
    }
}
type Body<'a> = fn(&mut Owner<'a>) -> Result<Output, ()>;

fn factory_control_bytes() -> Option<usize> {
    sum(&[
        size_of::<Owner<'_>>(),
        size_of::<Body<'_>>(),
        size_of::<Output>(),
        size_of::<Result<Result<Output, ()>, Error>>(),
        size_of::<(
            IndexedInvocationRequest<'_, MlxTensor>,
            &mut Callback<'_>,
            &Stream,
            OriginalIndexedResidencyFactory,
        )>(),
        size_of::<HostMetadataFunding>(),
    ])
}
fn owner_control_bytes() -> Option<usize> {
    OriginalIndexedResidencyInvocation::owner_control_bytes::<Owner<'_>, Output, (), Body<'_>>()
}
fn wrapper_control_bytes() -> Option<usize> {
    sum(&[
        size_of::<ChannelCall<'_>>(),
        size_of::<Option<IndexedBankSource>>(),
        size_of::<Option<eredu_nn::PreparedIndexedInvocationLoan<'_>>>(),
        size_of::<Result<Result<Output, ()>, Failure>>(),
        size_of::<Option<Result<Output, Error>>>(),
        size_of::<(
            IndexedInvocationRequest<'_, MlxTensor>,
            &mut Callback<'_>,
            &Stream,
        )>(),
        size_of::<CallbackAborted>(),
        size_of::<IndexedResidencyFactory>(),
    ])
}
fn ordinary_factory_control_bytes() -> Option<usize> {
    sum(&[
        size_of::<Owner<'_>>(),
        size_of::<Body<'_>>(),
        size_of::<Output>(),
        size_of::<Result<Result<Output, ()>, Error>>(),
        size_of::<(
            IndexedInvocationRequest<'_, MlxTensor>,
            &mut Callback<'_>,
            &Stream,
            OrdinaryIndexedResidencyFactory,
        )>(),
        size_of::<HostMetadataFunding>(),
    ])
}
fn reserve_wrapper(funding: &HostMetadataFunding) -> Result<(), Error> {
    funding
        .reserve_metadata(wrapper_control_bytes().ok_or_else(overflow)?)
        .map_err(Error::WorkspacePlanning)
}
fn run_ordinary_factory(
    callback: &mut Callback<'_>,
    request: IndexedInvocationRequest<'_, MlxTensor>,
    stream: &Stream,
    factory: OrdinaryIndexedResidencyFactory,
) -> Result<Result<Output, ()>, Error> {
    let funding = factory.funding().clone();
    factory.validate_request(&request)?;
    funding
        .reserve_metadata(
            ordinary_factory_control_bytes()
                .ok_or_else(overflow)?
                .checked_add(request.declaration.callback_control_bytes)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let invocation = factory.prepare(callback.movement(), stream)?;
    let mut owner = Owner {
        callback,
        funding: &funding,
    };
    let body: Body<'_> = Owner::body;
    invocation.run_with_owner(&mut owner, Owner::movement, stream, body)
}
fn run_factory(
    callback: &mut Callback<'_>,
    request: IndexedInvocationRequest<'_, MlxTensor>,
    stream: &Stream,
    factory: OriginalIndexedResidencyFactory,
) -> Result<Result<Output, ()>, Error> {
    let funding = factory.funding().clone();
    factory.validate_request(&request)?;
    funding
        .reserve_metadata(
            factory_control_bytes()
                .ok_or_else(overflow)?
                .checked_add(request.declaration.callback_control_bytes)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let invocation = factory.prepare(callback.movement(), stream)?;
    let mut owner = Owner {
        callback,
        funding: &funding,
    };
    let body: Body<'_> = Owner::body;
    invocation.run_with_owner(&mut owner, Owner::movement, stream, body)
}

struct ChannelCall<'a> {
    callback: &'a mut Callback<'a>,
    request: IndexedInvocationRequest<'a, MlxTensor>,
    stream: &'a Stream,
    used: bool,
    aborted: bool,
}
impl ChannelCall<'_> {
    fn run(&mut self, factory: IndexedResidencyFactory) -> Result<Output, Error> {
        if self.used {
            return Err(aborted());
        }
        self.used = true;
        reserve_wrapper(factory.funding())?;
        let result = match factory {
            IndexedResidencyFactory::Original(factory) => {
                run_factory(self.callback, self.request, self.stream, factory)
            }
            IndexedResidencyFactory::Ordinary(factory) => {
                run_ordinary_factory(self.callback, self.request, self.stream, factory)
            }
        }?;
        match result {
            Ok(output) => Ok(output),
            Err(()) => {
                self.aborted = true;
                Err(aborted())
            }
        }
    }
}
/// The absent loan is used only to inspect this exact closure's layout. It
/// constructs no model, native value, callback owner, or execution authority.
fn channel_adapter<'s, 'a: 's>(
    mut source: Option<&'s mut ChannelCall<'a>>,
) -> impl FnMut(IndexedResidencyFactory) -> Result<Output, Error> + 's + use<'s, 'a> {
    move |factory| source.as_deref_mut().ok_or_else(aborted)?.run(factory)
}
fn channel_control_bytes() -> Option<usize> {
    IndexedBankSource::request_region_control_bytes(size_of_val(&channel_adapter(None)))
}
impl MlxIndexedMovement {
    pub(crate) fn ordinary_chunk_control_bytes(
        census: eredu_runtime::expert::AddressableChunkCensus,
    ) -> Option<usize> {
        indexed_source::OrdinaryIndexedChunkSource::control_bytes(census)
    }
    pub(crate) fn ordinary_invocation_control_bytes() -> Option<usize> {
        ordinary_factory_control_bytes()?
            .checked_add(OrdinaryIndexedResidencyFactory::control_bytes()?)?
            .checked_add(
                indexed_source::OrdinaryIndexedResidencyInvocation::owner_control_bytes::<
                    Owner<'_>,
                    Output,
                    (),
                    Body<'_>,
                >()?,
            )?
            .checked_add(wrapper_control_bytes()?)?
            .checked_add(channel_control_bytes()?)
    }
    /// Exact weak-channel adapter charged to the installed request's metadata
    /// account before dispatch; the native region has a separate payer.
    pub(crate) fn request_channel_control_bytes() -> Option<usize> {
        channel_control_bytes()
    }
    /// Same concrete native workers used below. The installed request separately
    /// prices its channel; caller storage remains an independent source.
    pub(crate) fn invocation_control_bytes() -> Option<usize> {
        factory_control_bytes()?
            .checked_add(owner_control_bytes()?)?
            .checked_add(wrapper_control_bytes()?)
    }
}

pub(super) fn run(
    callback: &mut Callback<'_>,
    request: IndexedInvocationRequest<'_, MlxTensor>,
    source: Option<eredu_nn::PreparedIndexedInvocationLoan<'_>>,
    stream: &Stream,
) -> Result<Result<Output, ()>, Failure> {
    if let Some(mut source) = source {
        let funding = source.funding().clone();
        let slot = source
            .source_mut()
            .downcast_mut::<Option<OriginalIndexedResidencyFactory>>()
            .ok_or(Failure::MissingProducer)?;
        let factory = slot.take().ok_or(Failure::MissingProducer)?;
        if !factory.matches_funding(&funding) {
            return Err(Failure::Backend(identity()));
        }
        reserve_wrapper(factory.funding()).map_err(Failure::Backend)?;
        return run_factory(callback, request, stream, factory).map_err(Failure::Backend);
    }
    let binding = callback.movement().indexed_bank_source().cloned();
    if let Some(binding) = binding {
        let mut call = ChannelCall {
            callback,
            request,
            stream,
            used: false,
            aborted: false,
        };
        let mut invoke = channel_adapter(Some(&mut call));
        let result = binding.with_request_region(request, stream, &mut invoke);
        drop(invoke);
        if let Some(result) = result {
            if call.aborted {
                return Ok(Err(()));
            }
            if !call.used && result.is_ok() {
                return Err(Failure::MissingProducer);
            }
            return result.map(Ok).map_err(Failure::Backend);
        }
        if call.used {
            return Err(Failure::MissingProducer);
        }
    }
    indexed_source::require_ordinary().map_err(Failure::Backend)?;
    Ok(callback.run(None))
}
