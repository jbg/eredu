//! Exact host-only copies for the selected completion route consumer.
use super::*;
use std::{collections::TryReserveError, mem::{size_of,size_of_val}};
impl CommunicationRouteRealization {
    pub(crate) fn retention_copy_bytes(&self)->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<Result<Self,TryReserveError>>(),
            size_of::<Option<Group>>(),size_of::<Result<Option<Group>,TryReserveError>>(),
            size_of::<Option<eredu_runtime::RetainedCommunicationSource>>(),
            size_of::<&Self>(),size_of::<TryReserveError>(),size_of::<Option<usize>>()];
        let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?
            .checked_add(self.descriptor.retention_copy_bytes()?)?;
        match &self.group { Some(group)=>bytes.checked_add(group.retention_copy_bytes()?),None=>Some(bytes) }
    }
    pub(crate) fn try_copy_for_retention(&self)->Result<Self,TryReserveError> {
        Ok(Self { descriptor:self.descriptor.try_copy_for_retention()?,
            group:self.group.as_ref().map(Group::try_copy_for_retention).transpose()?,
            endpoint:self.endpoint,peer_rank:self.peer_rank,source:self.source.clone() })
    }
}
