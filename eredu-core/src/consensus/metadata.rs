//! Exact destinations of the shared protocol encoders and validators.
use super::ConsensusError;
use crate::{HostMetadataFunding,HostMetadataFundingError};
use std::mem::{size_of,size_of_val};

pub(crate) fn reserve_protocol(funding: Option<&HostMetadataFunding>, parts: &[usize]) -> Result<(),ConsensusError> {
    if let Some(funding) = funding {
        let bytes=parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
    }
    Ok(())
}
pub(crate) fn protocol_vec<T>(funding: Option<&HostMetadataFunding>, count: usize) -> Result<Vec<T>,ConsensusError> {
    let bytes=count.checked_mul(size_of::<T>()).ok_or(HostMetadataFundingError::Overflow)?;
    reserve_protocol(funding,&[bytes,size_of::<Vec<T>>(),size_of::<Result<Vec<T>,ConsensusError>>(),
        size_of::<std::collections::TryReserveError>(),size_of::<usize>()])?;
    let mut values=Vec::new();
    values.try_reserve_exact(count).map_err(|_|HostMetadataFundingError::Unavailable)?;
    Ok(values)
}

// Closed borrowed formatting: neither pass may mutate or allocate in Display.
// The output writer refuses drift instead of growing beyond the admitted bytes.
struct DiagnosticCount(usize);
impl std::fmt::Write for DiagnosticCount {
    fn write_str(&mut self,text:&str)->std::fmt::Result {
        self.0=self.0.checked_add(text.len()).ok_or(std::fmt::Error)?; Ok(())
    }
}
struct DiagnosticOutput(String);
impl std::fmt::Write for DiagnosticOutput {
    fn write_str(&mut self,text:&str)->std::fmt::Result {
        if text.len()>self.0.capacity()-self.0.len(){return Err(std::fmt::Error);}
        self.0.push_str(text); Ok(())
    }
}
pub(crate) fn protocol_string(funding:Option<&HostMetadataFunding>,arguments:std::fmt::Arguments<'_>)
    ->Result<String,HostMetadataFundingError> {
    let Some(funding)=funding else{return Ok(std::fmt::format(arguments))};
    let mut count=DiagnosticCount(0);
    std::fmt::write(&mut count,arguments).map_err(|_|HostMetadataFundingError::Unavailable)?;
    let parts=[count.0,size_of::<DiagnosticCount>(),size_of::<DiagnosticOutput>(),
        size_of::<std::fmt::Arguments<'_>>(),size_of::<std::fmt::Result>(),
        size_of::<Result<String,HostMetadataFundingError>>(),size_of::<std::collections::TryReserveError>(),
        HostMetadataFunding::reservation_control_bytes()];
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(HostMetadataFundingError::Overflow)?)?;
    let mut output=DiagnosticOutput(String::new());
    output.0.try_reserve_exact(count.0).map_err(|_|HostMetadataFundingError::Unavailable)?;
    std::fmt::write(&mut output,arguments).map_err(|_|HostMetadataFundingError::Unavailable)?;
    if output.0.len()!=count.0{return Err(HostMetadataFundingError::Unavailable);}
    Ok(output.0)
}
