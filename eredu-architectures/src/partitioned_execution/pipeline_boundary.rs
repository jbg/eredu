//! Destinations lent by the actual pipeline model invocation.
use eredu_nn::{workspace::WorkspaceContext,Error};
pub(super) fn vector<T>(context:Option<&WorkspaceContext>,count:usize)->Result<Vec<T>,Error>{
    match context{Some(context)=>context.metadata_vec(count),None=>Ok(Vec::with_capacity(count))}
}
pub(super) fn value<T>(context:Option<&WorkspaceContext>,role:&str,tensor:T)
    ->Result<eredu_runtime::ArchitectureBoundaryValue<T>,Error>{
    match context{
        Some(context)=>eredu_runtime::ArchitectureBoundaryValue::new_with_metadata(role,tensor,context),
        None=>eredu_runtime::ArchitectureBoundaryValue::new(role,tensor).map_err(Error::backend_retained_source),
    }
}
pub(super) fn source<E:std::error::Error+Send+Sync+'static>(context:Option<&WorkspaceContext>,cause:E)->Error{
    match context{Some(context)=>context.metadata_source(cause),None=>Error::backend_retained_source(cause)}
}
pub(super) fn error(context:Option<&WorkspaceContext>,args:std::fmt::Arguments<'_>)->Error{
    match context{Some(context)=>context.metadata_error(args),None=>Error::backend(args)}
}

pub(super) fn text(context:Option<&WorkspaceContext>,args:std::fmt::Arguments<'_>)->Result<String,Error>{
    match context{Some(context)=>context.metadata_string(args),None=>Ok(args.to_string())}
}

// Keep the original typed cause and its account when a checked partition
// producer refuses. Static stage names require no diagnostic string producer.
#[derive(Debug)]
struct StageFailure {
    stage: &'static str,
    equation: Option<(usize, String)>,
    inspection: Option<Error>,
    cause: Error,
    _funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}
impl std::fmt::Display for StageFailure {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        write!(f,"pipeline {}: {}",self.stage,self.cause)?;
        if let Some((index,description))=&self.equation {
            write!(f,"; first incomplete equation {index}: {description}")?;
        }
        if let Some(inspection)=&self.inspection {
            write!(f,"; quote inspection refused: {inspection}")?;
        }
        Ok(())
    }
}
impl std::error::Error for StageFailure {
    fn source(&self)->Option<&(dyn std::error::Error+'static)>{Some(&self.cause)}
}
pub(super) fn stage(context: Option<&WorkspaceContext>, stage: &'static str, cause: Error) -> Error {
    let Some(context) = context else { return cause; };
    if let Err(refusal) = context.charge_metadata(std::mem::size_of::<(
        StageFailure, &WorkspaceContext, &'static str, Result<(), Error>, Error,
    )>()) { return refusal.into(); }
    context.metadata_source(StageFailure { stage, equation: None, inspection: None, cause,
        _funding: context.metadata_funding() })
}

/// The error enclosure is admitted before entering the unit. On an aborted
/// quote it can retain both the original failure and a refused inspection
/// without another allocation callback after that refusal.
pub(super) struct UnitFailure<'a> { context: Option<&'a WorkspaceContext> }
impl<'a> UnitFailure<'a> {
    pub(super) fn prepare(context: Option<&'a WorkspaceContext>) -> Result<Self, Error> {
        use std::mem::{size_of,size_of_val};
        use eredu_nn::workspace::{WorkspaceOperation,WorkspaceTraceReport,WorkspaceMetadataError};
        if let Some(context)=context {
            let controls=[
                Error::retained_source_construction_bytes::<StageFailure>().ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<Self>(),size_of::<StageFailure>(),size_of::<Result<Self,Error>>(),
                size_of::<Option<&WorkspaceContext>>(),size_of::<&WorkspaceContext>(),
                size_of::<Error>()*3,size_of::<Result<WorkspaceTraceReport,Error>>(),
                size_of::<WorkspaceTraceReport>(),size_of::<Option<usize>>(),
                size_of::<Option<(usize,String)>>(),size_of::<String>(),size_of::<Result<String,Error>>(),
                size_of::<std::fmt::Arguments<'_>>(),size_of::<(&WorkspaceContext,&WorkspaceOperation)>(),
                size_of::<WorkspaceOperation>(),size_of::<(usize,WorkspaceOperation)>(),
                size_of::<Option<(usize,WorkspaceOperation)>>(),
                size_of::<std::iter::Enumerate<std::vec::IntoIter<WorkspaceOperation>>>(),
                size_of::<std::slice::Iter<WorkspaceOperation>>(),
                size_of::<Option<&usize>>(),
                size_of::<std::slice::Iter<eredu_nn::workspace::WorkspaceLayout>>(),
                size_of::<Option<&(dyn std::error::Error+'static)>>(),
                size_of::<Option<&eredu_nn::workspace::WorkspaceMetadataError>>(),
                size_of::<Option<&eredu_core::BackendFailure>>(),
                size_of::<Option<&eredu_nn::workspace::HostMetadataFundingError>>(),
                size_of::<bool>(),size_of::<usize>(),size_of::<Option<Error>>(),
                size_of::<(Self,Error)>(),size_of::<&WorkspaceOperation>(),
            ];
            context.charge_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?)?;
        }
        Ok(Self{context})
    }
    pub(super) fn retain(self,cause:Error)->Error {
        use eredu_nn::workspace::{WorkspaceDtype,WorkspaceMetadataError};
        let Some(context)=self.context else{return cause};
        let mut source:Option<&(dyn std::error::Error+'static)>=Some(&cause);
        while let Some(current)=source {
            if current.is::<eredu_nn::workspace::HostMetadataFundingError>()
                || matches!(current.downcast_ref::<WorkspaceMetadataError>(),Some(
                    WorkspaceMetadataError::Capacity{..}|WorkspaceMetadataError::Funding(_)|WorkspaceMetadataError::Overflow))
                || current.downcast_ref::<eredu_core::BackendFailure>().is_some_and(|error|
                    error.kind()==eredu_core::BackendFailureKind::ResourceExhausted) {
                return cause;
            }
            source=current.source();
        }
        let mut equation=None;
        let mut inspection=None;
        match context.finish_report(&[]) {
            Ok(report)=>{
                // A zero-output unpriced store can precede the actual scalar loss.
                // Prefer the first missing floating result across the entire aborted trace.
                let missing=report.operations.iter().position(|operation|operation.outputs.iter().any(|output|
                    output.dtype()==WorkspaceDtype::Float32&&output.representation().is_none()))
                    .or_else(||report.unpriced_operations.first().copied());
                for (index,operation) in report.operations.into_iter().enumerate() {
                    if Some(index)==missing {
                        match context.metadata_string(format_args!("{operation:?}")) {
                            Ok(description)=>equation=Some((index,description)),
                            Err(error)=>inspection=Some(error),
                        }
                        break;
                    }
                }
            }
            Err(error)=>inspection=Some(error),
        }
        // Both allocations and all fixed transports were charged before the
        // unit call. Diagnostic text uses the same paid aborted trace and retains its account.
        Error::backend_retained_source(StageFailure{stage:"unit forward",equation,inspection,cause,
            _funding:context.metadata_funding()})
    }
}

#[cfg(test)]
#[path="pipeline_boundary_tests.rs"]
mod tests;
