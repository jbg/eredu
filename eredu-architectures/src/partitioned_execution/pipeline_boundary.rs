//! Destinations lent by the actual pipeline model invocation.
use eredu_nn::{workspace::WorkspaceContext,Error};
pub(super) fn vector<T>(context:Option<&WorkspaceContext>,count:usize)->Result<Vec<T>,Error>{
    match context{Some(context)=>context.metadata_vec(count),None=>Ok(Vec::with_capacity(count))}
}
pub(super) fn value<T>(context:Option<&WorkspaceContext>,role:&str,tensor:T)
    ->Result<eredu_runtime::ArchitectureBoundaryValue<T>,Error>{
    match context{
        Some(context)=>eredu_runtime::ArchitectureBoundaryValue::new_with_metadata(role,tensor,context),
        None=>eredu_runtime::ArchitectureBoundaryValue::new(role,tensor).map_err(Error::backend_source),
    }
}
pub(super) fn source<E:std::error::Error+Send+Sync+'static>(context:Option<&WorkspaceContext>,cause:E)->Error{
    match context{Some(context)=>context.metadata_source(cause),None=>Error::backend_source(cause)}
}
pub(super) fn error(context:Option<&WorkspaceContext>,args:std::fmt::Arguments<'_>)->Error{
    match context{Some(context)=>context.metadata_error(args),None=>Error::backend(args)}
}

pub(super) fn text(context:Option<&WorkspaceContext>,args:std::fmt::Arguments<'_>)->Result<String,Error>{
    match context{Some(context)=>context.metadata_string(args),None=>Ok(args.to_string())}
}
