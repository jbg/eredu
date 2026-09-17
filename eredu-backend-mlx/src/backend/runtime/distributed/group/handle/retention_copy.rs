//! Host-only retention of the actual native/logical group declaration.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::{size_of, size_of_val}};

fn vector_bytes<T>(length:usize)->Option<usize> {
    let parts=[Layout::array::<T>(length).ok()?.size(),size_of::<Vec<T>>(),
        size_of::<std::result::Result<Vec<T>,TryReserveError>>(),
        size_of::<std::result::Result<(),TryReserveError>>(),
        size_of::<(&[T],usize)>(),size_of::<std::slice::Iter<'_,T>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
fn vector<T>(length:usize)->std::result::Result<Vec<T>,TryReserveError> {
    let mut output=Vec::new(); output.try_reserve_exact(length)?; Ok(output)
}
fn copy<T:Copy>(values:&[T])->std::result::Result<Vec<T>,TryReserveError> {
    let mut output=vector(values.len())?; output.extend_from_slice(values); Ok(output)
}
impl Group {
    /// Includes all actual nested destinations, without querying or submitting
    /// native work. Native Group and transport-stream clones are Rc aliases.
    pub(crate) fn retention_copy_bytes(&self)->Option<usize> {
        let parts=[size_of::<Self>(),size_of::<std::result::Result<Self,TryReserveError>>(),
            size_of::<Option<LogicalSubgroup>>(),size_of::<LogicalSubgroup>(),
            size_of::<Option<ManifestGroupContract>>(),size_of::<ManifestGroupContract>(),
            size_of::<Option<Vec<LogicalRoute>>>(),size_of::<LogicalRoute>(),
            size_of::<std::result::Result<LogicalSubgroup,TryReserveError>>(),
            size_of::<std::result::Result<ManifestGroupContract,TryReserveError>>(),
            size_of::<native::Group>(),size_of::<Rc<OnceCell<Stream>>>(),
            size_of::<Option<native::RetainedGroupBuffer>>(),native::RetainedGroupBuffer::clone_control_bytes(),
            size_of::<Option<eredu_runtime::RetainedCommunicationSource>>(),
            size_of::<Option<super::super::super::topology::original_source::parallel::OriginalParallelBinding>>(),
            size_of::<Option<super::super::super::topology::original_source::control::OriginalControlBinding>>(),
            size_of::<(&Self,usize)>(),size_of::<TryReserveError>(),size_of::<Option<usize>>()];
        let mut bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?;
        if self.original_control_request.is_some() {
            bytes=bytes.checked_add(super::super::super::topology::original_source::control::OriginalParallelControlProjection::alias_control_bytes()?)?;
        }
        if let Some(contract)=&self.contract {
            bytes=bytes.checked_add(contract.requirements.retention_copy_bytes()?)?;
        }
        if let Some(logical)=&self.logical {
            bytes=bytes.checked_add(vector_bytes::<usize>(logical.global_ranks.len())?)?;
            if let Some(routes)=&logical.routes {
                bytes=bytes.checked_add(vector_bytes::<LogicalRoute>(routes.len())?)?;
                for route in routes {
                    bytes=bytes.checked_add(vector_bytes::<Option<usize>>(route.exchanges.len())?)?;
                }
            }
        }
        Some(bytes)
    }
    /// The completion source consumer prepays `retention_copy_bytes` and keeps
    /// its account alive while this value enters the existing native retention.
    pub(crate) fn try_copy_for_retention(&self)->std::result::Result<Self,TryReserveError> {
        let contract=match &self.contract {
            Some(value)=>Some(ManifestGroupContract { id:value.id,
                requirements:value.requirements.try_copy_for_retention()? }),
            None=>None,
        };
        let logical=match &self.logical {
            Some(value)=>{
                let global_ranks=copy(&value.global_ranks)?;
                let routes=match &value.routes {
                    Some(values)=>{
                        let mut copied=vector(values.len())?;
                        for route in values { copied.push(LogicalRoute {
                            source_rank:route.source_rank,exchanges:copy(&route.exchanges)? }); }
                        Some(copied)
                    },
                    None=>None,
                };
                Some(LogicalSubgroup { global_ranks,rank:value.rank,routes,
                    world_collective_wave:value.world_collective_wave })
            },
            None=>None,
        };
        Ok(Self { native:self.native.clone(),retained_buffer:self.retained_buffer.clone(),transport_stream:Rc::clone(&self.transport_stream),
            logical,contract,completion:self.completion,source:self.source.clone(),original_parallel:self.original_parallel.clone(),original_control:self.original_control.clone(),original_control_request:self.original_control_request.clone() })
    }
}
