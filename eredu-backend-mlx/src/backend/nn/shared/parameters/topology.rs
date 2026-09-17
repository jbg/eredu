//! Immutable native parameter table shared by ordinary and paid constructors.
use super::*;
use eredu_core::{SharedStorageOwner,SharedStorageRetirement};
use eredu_nn::workspace::{WorkspaceContext,WorkspaceMetadataFunding,WorkspaceMetadataFundingError};
use std::{alloc::Layout,mem::{size_of,size_of_val},sync::Arc};

pub(in crate::backend::nn::shared) enum TopologyKeys<'a> {
    Ordinary(std::collections::btree_map::Keys<'a,String,ParameterSpec>),
    Prepared(std::slice::Iter<'a,(&'static str,ParameterSpec)>),
}
impl<'a> Iterator for TopologyKeys<'a> {
    type Item=&'a str;
    fn next(&mut self)->Option<Self::Item>{match self {
        Self::Ordinary(keys)=>keys.next().map(String::as_str),
        Self::Prepared(rows)=>rows.next().map(|(name,_)|*name),
    }}
    fn size_hint(&self)->(usize,Option<usize>){let n=self.len();(n,Some(n))}
}
impl ExactSizeIterator for TopologyKeys<'_> {
    fn len(&self)->usize{match self{Self::Ordinary(keys)=>keys.len(),Self::Prepared(rows)=>rows.len()}}
}
/// Backend-owned table loans only; they grant no parameter or graph authority.
pub(in crate::backend::nn::shared) trait NativeParameterTopology {
    fn len(&self)->usize;
    fn get(&self,name:&str)->Option<&ParameterSpec>;
    fn keys(&self)->TopologyKeys<'_>;
    fn contains_key(&self,name:&str)->bool{self.get(name).is_some()}
}
impl NativeParameterTopology for BTreeMap<String,ParameterSpec> {
    fn len(&self)->usize{BTreeMap::len(self)}
    fn get(&self,name:&str)->Option<&ParameterSpec>{BTreeMap::get(self,name)}
    fn keys(&self)->TopologyKeys<'_>{TopologyKeys::Ordinary(BTreeMap::keys(self))}
}
#[derive(Debug,Clone)]
pub(in crate::backend::nn::shared) enum NamedParameterTopology {
    Ordinary(BTreeMap<String,ParameterSpec>),
    Prepared(PreparedParameterTopology),
}
impl NativeParameterTopology for NamedParameterTopology {
    fn len(&self)->usize{match self{Self::Ordinary(map)=>map.len(),Self::Prepared(table)=>table.len()}}
    fn get(&self,name:&str)->Option<&ParameterSpec>{match self{Self::Ordinary(map)=>map.get(name),Self::Prepared(table)=>table.get(name)}}
    fn keys(&self)->TopologyKeys<'_>{match self{Self::Ordinary(map)=>TopologyKeys::Ordinary(map.keys()),Self::Prepared(table)=>table.keys()}}
}
#[derive(Debug,thiserror::Error)]
enum Cause {
    #[error("prepared parameter table differs from its exact native fields")]
    Identity,
    #[error("prepared parameter table extent overflowed")]
    Overflow,
    #[error("prepared parameter table source: {0}")]
    Source(#[source]ParameterSourceError),
    #[error("prepared parameter table funding: {0}")]
    Funding(#[source]WorkspaceMetadataFundingError),
    #[error("prepared parameter table destination: {0}")]
    Allocation(#[source]std::collections::TryReserveError),
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct Failure { #[source]cause:Cause, funding:WorkspaceMetadataFunding }
fn failure(cause:Cause,funding:&WorkspaceMetadataFunding)->ComputeError {
    ComputeError::backend_retained_source(Failure{cause,funding:funding.clone()})
}
struct Rows { rows:Vec<(&'static str,ParameterSpec)>, funding:WorkspaceMetadataFunding }
impl SharedStorageRetirement for Rows {
    fn retire(self:Arc<Self>){drop(Arc::into_inner(self));}
}
/// Clones share the immutable paid table, including its complete retirement.
#[derive(Debug,Clone)]
pub(in crate::backend::nn::shared) struct PreparedParameterTopology(SharedStorageOwner<Rows>);
impl NativeParameterTopology for PreparedParameterTopology {
    fn len(&self)->usize{self.0.rows.len()}
    fn get(&self,name:&str)->Option<&ParameterSpec>{
        self.0.rows.binary_search_by(|(key,_)|key.cmp(&name)).ok().map(|index|&self.0.rows[index].1)
    }
    fn keys(&self)->TopologyKeys<'_>{TopologyKeys::Prepared(self.0.rows.iter())}
}
pub(in crate::backend::nn::shared) struct PreparedTopologyBuilder {
    rows:Vec<(&'static str,ParameterSpec)>,
    limit:usize,
    failed:bool,
    funding:WorkspaceMetadataFunding,
}
impl PreparedParameterTopology {
    /// The caller separately supplies already-paid owned ParameterSpec values.
    /// This query covers the fixed directory, shared owner and direct verifier.
    pub(in crate::backend::nn::shared) fn control_bytes(rows:usize)->Option<usize>{
        let parts=[size_of::<Self>(),size_of::<NamedParameterTopology>(),size_of::<PreparedTopologyBuilder>(),
            size_of::<Rows>(),size_of::<SharedStorageOwner<Rows>>(),
            size_of::<Result<Self,ComputeError>>(),size_of::<Result<PreparedTopologyBuilder,ComputeError>>(),
            size_of::<(usize,&'static str,ParameterSpec)>(),size_of::<TopologyKeys<'_>>(),
            Layout::array::<(&'static str,ParameterSpec)>(rows).ok()?.size(),
            WorkspaceContext::metadata_arc_bytes::<Rows>()?,
            sources::binding_visit_control_bytes()?,
            ComputeError::retained_source_control_bytes::<Failure>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(in crate::backend::nn::shared) fn prepare(rows:usize,funding:&WorkspaceMetadataFunding)
        ->Result<PreparedTopologyBuilder,ComputeError>{
        let controls=Self::control_bytes(rows).ok_or_else(||failure(Cause::Overflow,funding))?;
        funding.reserve_metadata(controls).map_err(|cause|failure(Cause::Funding(cause),funding))?;
        let mut values=Vec::new();values.try_reserve_exact(rows).map_err(|cause|failure(Cause::Allocation(cause),funding))?;
        Ok(PreparedTopologyBuilder{rows:values,limit:rows,failed:false,funding:funding.clone()})
    }
    pub(super) fn validate<M:NativeRetainedValues>(&self,module:&M)->Result<(),ComputeError>{
        struct Check;
        impl<'a> ParameterSourceVisitor<'a,MlxTensor> for Check {
            fn parameter(&mut self,_:ParameterMetadataView<'a>,_:&'a MlxTensor){}
            fn retained(&mut self,_:&'a MlxTensor){}
        }
        // The same source verifier checks exact names, count and unique physical
        // fields for both table types before any parameter loan is published.
        visit_module_parameter_sources(module,self,&mut Check)
            .map_err(|cause|failure(Cause::Source(cause),&self.0.funding))
    }
}
impl PreparedTopologyBuilder {
    pub(in crate::backend::nn::shared) fn push(&mut self,name:&'static str,spec:ParameterSpec)->Result<(),ComputeError>{
        if self.failed||name.is_empty()||self.rows.len()==self.limit||self.rows.last().is_some_and(|(key,_)|*key>=name){
            self.failed=true;return Err(failure(Cause::Identity,&self.funding));
        }
        self.rows.push((name,spec));Ok(())
    }
    pub(in crate::backend::nn::shared) fn finish(self)->Result<PreparedParameterTopology,ComputeError>{
        if self.failed||self.rows.len()!=self.limit{return Err(failure(Cause::Identity,&self.funding));}
        Ok(PreparedParameterTopology(SharedStorageOwner::new(Rows{rows:self.rows,funding:self.funding})))
    }
}
