//! Existing-only physical pins for the exact immutable communicator table.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::working_memory::{ExistingStoragePinLayout,WorkingMemoryPool,WorkingMemoryStorage};

/// Minted only after every actual nonzero table buffer has been pinned. The
/// source alias carries the actual accounting owner; this token supplies no
/// caller-controlled identity or byte credit.
#[derive(Clone,Copy)]
pub(super) struct RegisteredBuffers(());
struct Custody {
    _pins:WorkingMemoryStorage<StorageIdentity>,
    // Pin keys/registration retire before their shared shell and H account.
    _funding:WorkspaceMetadataFunding,
}
fn groups(actual:&ParallelCommunicators)->impl Iterator<Item=&Group> {
    std::iter::once(&actual.control_world)
        .chain(actual.groups.values().filter_map(|group|group.native.as_ref()))
        .chain(actual.routes.values().filter_map(|route|route.group.as_ref()))
}

pub(super) fn pin(actual:&ParallelCommunicators,source:&RetainedCommunicationSource,
    funding:&WorkspaceMetadataFunding,pool:&WorkingMemoryPool)
    ->Result<(RetainedCommunicationSource,RegisteredBuffers),Error> {
    let overflow=||Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow);
    let population=1usize.checked_add(actual.groups.len())
        .and_then(|n|n.checked_add(actual.routes.len())).ok_or_else(overflow)?;
    let actual_groups=groups(actual);
    let controls=[size_of::<(&ParallelCommunicators,&RetainedCommunicationSource,
            &WorkspaceMetadataFunding,&WorkingMemoryPool)>(),
        size_of::<(RetainedCommunicationSource,RegisteredBuffers)>(),
        size_of::<Result<(RetainedCommunicationSource,RegisteredBuffers),Error>>(),
        size_of::<ExistingStoragePinLayout<StorageIdentity>>(),
        size_of::<Result<ExistingStoragePinLayout<StorageIdentity>,eredu_runtime::working_memory::WorkingMemoryError>>(),
        size_of::<Result<WorkingMemoryStorage<StorageIdentity>,eredu_runtime::working_memory::WorkingMemoryError>>(),
        size_of::<Vec<(StorageIdentity,u64)>>(),size_of::<(StorageIdentity,u64)>(),
        size_of::<Option<&safemlx::distributed::RetainedGroupBuffer>>(),
        size_of::<(&safemlx::distributed::RetainedGroupBuffer,&safemlx::distributed::Group)>(),
        size_of::<Option<RetainedCommunicationSource>>(),size_of::<Custody>(),
        size_of::<usize>()*3,size_of::<u64>(),size_of::<Option<usize>>(),
        size_of::<bool>()*3,size_of_val(&actual_groups),
        HostPreparationAuthority::retention_bytes::<Custody>().ok_or_else(overflow)?,
        failure_control_bytes().ok_or_else(overflow)?];
    funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
    let layout=ExistingStoragePinLayout::<StorageIdentity>::new(population)
        .map_err(|cause|failure(Cause::Allocator(cause),source,funding))?;
    funding.reserve_metadata(layout.requested_bytes()).map_err(Error::WorkspacePlanning)?;
    if !source.same_source(&actual.source) {
        return Err(failure(Cause::Identity,source,funding));
    }
    // The existing counted vector producer funds its exact payload and failure
    // controls. No native query, clone, registration, or inferred shape runs.
    let mut rows=funding.metadata_vec::<(StorageIdentity,u64)>(population).map_err(Error::Neural)?;
    for group in actual_groups {
        let buffer=group.retained_buffer().ok_or_else(||failure(Cause::Resource,source,funding))?;
        if !buffer.is_for(group.native_group()) || rows.len()==population {
            return Err(failure(Cause::Identity,source,funding));
        }
        let bytes=u64::try_from(buffer.bytes()).map_err(|_|overflow())?;
        if bytes!=0 {rows.push((StorageIdentity::GroupBuffer(buffer.identity()),bytes));}
    }
    // The common atomic worker rejects missing keys and contradictory capacity;
    // it adds owners to existing registrations without charging physical bytes.
    let pins=layout.construct(pool,rows)
        .map_err(|cause|failure(Cause::Allocator(cause),source,funding))?;
    let custody=HostPreparationAuthority::retain(Custody{_pins:pins,_funding:funding.clone()});
    let retained=source.with_host_preparation(&custody)
        .ok_or_else(||failure(Cause::Identity,source,funding))?;
    Ok((retained,RegisteredBuffers(())))
}
