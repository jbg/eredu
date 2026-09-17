//! Canonical boundary headers and finite source-retaining frame destinations.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding,WorkspaceMetadataFundingError};
use std::{alloc::Layout,collections::TryReserveError,mem::{size_of,size_of_val}};

/// One checked header layout. Both ordinary and prepared paths invoke its
/// identical writer; no shape vector or second binary protocol is created.
enum HeaderShape<'a> {
    Declared(&'a [BoundaryDimensionContract]),
    Actual(&'a [i32]),
}
impl HeaderShape<'_> {
    fn len(&self)->usize{match self{Self::Declared(v)=>v.len(),Self::Actual(v)=>v.len()}}
    fn extent(&self,index:usize)->Option<u64>{match self{
        Self::Declared(values)=>match values.get(index)?{
            BoundaryDimensionContract::Fixed(value)=>u64::try_from(*value).ok(),
            BoundaryDimensionContract::Variable{..}=>None,
        },
        Self::Actual(values)=>u64::try_from(*values.get(index)?).ok(),
    }}
}
pub(super) struct Header<'a> {
    route:CommunicationRouteId,
    schema:&'a str,
    role:&'a str,
    shape:HeaderShape<'a>,
    ordinal:u32,
    schema_len:u32,
    role_len:u32,
    rank:u32,
    payload_bytes:u64,
    dtype:u8,
    bytes:usize,
}
impl<'a> Header<'a> {
    fn source_control_bytes()->Option<usize>{
        let parts=[size_of::<Header<'_>>(),size_of::<HeaderShape<'_>>(),
            size_of::<(CommunicationRouteId,&str,usize,&BoundaryRoleContract,HeaderShape<'_>)>(),
            size_of::<(&[i32],&TensorDtype)>(),size_of::<std::ops::Range<usize>>(),
            size_of::<std::iter::Zip<std::slice::Iter<'_,i32>,std::slice::Iter<'_,BoundaryDimensionContract>>>(),
            size_of::<(usize,usize,u64,u32,u32,u32,u32,u8)>(),
            size_of::<Result<Header<'_>,CommunicationManifestError>>(),
            size_of::<Option<usize>>(),size_of::<Option<u64>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }

    pub(super) fn new(route:CommunicationRouteId,schema:&'a str,ordinal:usize,role:&'a BoundaryRoleContract)
        ->Result<Self,CommunicationManifestError>{
        Self::with_shape(route,schema,ordinal,role,HeaderShape::Declared(&role.shape))
    }
    fn actual(route:CommunicationRouteId,schema:&'a str,ordinal:usize,role:&'a BoundaryRoleContract,
        shape:&'a [i32],dtype:&TensorDtype)->Result<Self,CommunicationManifestError>{
        let invalid=||CommunicationManifestError::InvalidBoundaryContract;
        if dtype!=&role.dtype || shape.len()!=role.shape.len(){return Err(invalid());}
        for (actual,declared) in shape.iter().zip(&role.shape){
            let actual=usize::try_from(*actual).map_err(|_|invalid())?;
            if match declared {
                BoundaryDimensionContract::Fixed(expected)=>actual!=*expected,
                BoundaryDimensionContract::Variable{maximum}=>actual>*maximum,
            } {return Err(invalid());}
        }
        Self::with_shape(route,schema,ordinal,role,HeaderShape::Actual(shape))
    }
    fn with_shape(route:CommunicationRouteId,schema:&'a str,ordinal:usize,role:&'a BoundaryRoleContract,
        shape:HeaderShape<'a>)->Result<Self,CommunicationManifestError>{
        let invalid=||CommunicationManifestError::InvalidBoundaryContract;
        let payload_elements=(0..shape.len()).try_fold(1usize,|total,index|
            total.checked_mul(usize::try_from(shape.extent(index)?).ok()?)).ok_or_else(invalid)?;
        let payload_bytes=payload_elements.checked_mul(tensor_dtype_width(&role.dtype).ok_or_else(invalid)?)
            .and_then(|bytes|u64::try_from(bytes).ok()).ok_or_else(invalid)?;
        let ordinal=u32::try_from(ordinal).map_err(|_|invalid())?;
        let schema_len=u32::try_from(schema.len()).map_err(|_|invalid())?;
        let role_len=u32::try_from(role.role.len()).map_err(|_|invalid())?;
        let rank=u32::try_from(shape.len()).map_err(|_|invalid())?;
        let bytes=8usize.checked_add(size_of::<u16>()).and_then(|n|n.checked_add(size_of::<u64>()))
            .and_then(|n|n.checked_add(size_of::<u32>())).and_then(|n|n.checked_add(size_of::<u8>()))
            .and_then(|n|n.checked_add(3*size_of::<u32>())).and_then(|n|n.checked_add(size_of::<u64>()))
            .and_then(|n|n.checked_add(schema.len())).and_then(|n|n.checked_add(role.role.len()))
            .and_then(|n|shape.len().checked_mul(size_of::<u64>()).and_then(|shape|n.checked_add(shape)))
            .ok_or_else(invalid)?;
        Ok(Self{route,schema,role:&role.role,shape,ordinal,schema_len,role_len,rank,payload_bytes,
            dtype:dtype_tag(&role.dtype)?,bytes})
    }
    pub(super) fn bytes(&self)->usize{self.bytes}
    pub(super) fn write(self,out:&mut Vec<u8>){
        debug_assert!(out.is_empty()&&out.capacity()>=self.bytes);
        out.extend_from_slice(b"EREDUBND");out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&self.route.value().to_le_bytes());out.extend_from_slice(&self.ordinal.to_le_bytes());
        out.push(self.dtype);out.extend_from_slice(&self.schema_len.to_le_bytes());out.extend_from_slice(self.schema.as_bytes());
        out.extend_from_slice(&self.role_len.to_le_bytes());out.extend_from_slice(self.role.as_bytes());
        out.extend_from_slice(&self.rank.to_le_bytes());
        for index in 0..self.shape.len(){out.extend_from_slice(&self.shape.extent(index).expect("checked header shape").to_le_bytes());}
        out.extend_from_slice(&self.payload_bytes.to_le_bytes());
        debug_assert_eq!(out.len(),self.bytes);
    }
}

/// A finite framing attempt failed before native submission.
#[derive(Debug,thiserror::Error)]
pub enum PreparedBoundaryFrameCause {
    /// The actual route/role/shape did not match its retained declaration.
    #[error("prepared boundary frame differs from its retained route contract")]
    Contract,
    /// Exact frame storage could not be charged to the supplied account.
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    /// A prepaid vector could not be allocated.
    #[error("prepared boundary frame destination allocation failed: {0}")]
    Capacity(#[source] TryReserveError),
}
/// Inline error custody keeps the exact declaration and account without another
/// allocation. A backend translation must preserve this original source.
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
pub struct PreparedBoundaryFrameError {
    #[source]
    cause:PreparedBoundaryFrameCause,
    _source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
/// Canonical in-band headers and logical tensors under one exact route source.
/// This prices host destinations only; native graph/backing/completion and role
/// admission remain backend-owned and cannot be inferred from header lengths.
#[derive(Debug)]
pub struct PreparedBoundaryFrames<T> {
    values:Vec<crate::RoleExactBoundaryValue<T>>,
    route:CommunicationRouteId,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl<T> PreparedBoundaryFrames<T> {
    /// Exact selected directed route.
    pub fn route(&self)->CommunicationRouteId{self.route}
    /// Immutable canonical frames, preserving architecture role order.
    pub fn values(&self)->&[crate::RoleExactBoundaryValue<T>]{&self.values}
    /// Original declaration source; equal descriptors do not replace it.
    pub fn source(&self)->&RetainedCommunicationSource{&self.source}
    /// Account which owns the finite frame and error destinations.
    pub fn funding(&self)->&WorkspaceMetadataFunding{&self.funding}
    /// Move the paid frames and custody into native completion preparation.
    /// The receiver retains both source and funding through all escaped frames.
    pub fn into_parts(self)->(Vec<crate::RoleExactBoundaryValue<T>>,RetainedCommunicationSource,WorkspaceMetadataFunding){
        (self.values,self.source,self.funding)
    }
}
impl RetainedCommunicationSource {
    /// Validate invocation roles against this exact retained route and create
    /// their canonical headers in charged destinations. Every refusal keeps
    /// the same source/funding; no inference or native submission occurs here.
    pub fn prepare_boundary_frames<T>(&self,route:CommunicationRouteId,
        actual_roles:&[BoundaryRoleContract],values:Vec<T>,funding:&WorkspaceMetadataFunding)
        ->Result<PreparedBoundaryFrames<T>,PreparedBoundaryFrameError>
    {
        let fail=|cause|PreparedBoundaryFrameError{cause,_source:self.clone(),_funding:funding.clone()};
        let overflow=||fail(PreparedBoundaryFrameCause::Funding(WorkspaceMetadataFundingError::Overflow));
        let controls=[size_of::<PreparedBoundaryFrames<T>>(),size_of::<Result<PreparedBoundaryFrames<T>,PreparedBoundaryFrameError>>(),
            size_of::<PreparedBoundaryFrameError>(),size_of::<PreparedBoundaryFrameCause>(),Header::source_control_bytes().ok_or_else(overflow)?,
            size_of::<Result<Header<'_>,CommunicationManifestError>>(),size_of::<Vec<T>>(),
            size_of::<std::vec::IntoIter<T>>(),size_of::<crate::RoleExactBoundaryValue<T>>(),
            size_of::<Vec<u8>>(),size_of::<Result<(),TryReserveError>>(),size_of::<Layout>(),
            size_of::<(&Self,CommunicationRouteId,&[BoundaryRoleContract],&WorkspaceMetadataFunding)>(),
            size_of::<(usize,usize,u32,u64)>(),size_of::<Option<usize>>()];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add).ok_or_else(overflow)?)
            .map_err(|cause|fail(PreparedBoundaryFrameCause::Funding(cause)))?;
        let descriptor=self.manifest().routes().iter().find(|candidate|candidate.id()==route)
            .ok_or_else(||fail(PreparedBoundaryFrameCause::Contract))?;
        let contract=descriptor.boundary_contract().ok_or_else(||fail(PreparedBoundaryFrameCause::Contract))?;
        if values.len()!=contract.roles.len(){return Err(fail(PreparedBoundaryFrameCause::Contract));}
        contract.validate_actual_roles(actual_roles).map_err(|_|fail(PreparedBoundaryFrameCause::Contract))?;
        funding.reserve_metadata(Layout::array::<crate::RoleExactBoundaryValue<T>>(values.len()).map_err(|_|overflow())?.size())
            .map_err(|cause|fail(PreparedBoundaryFrameCause::Funding(cause)))?;
        let mut frames=Vec::new();frames.try_reserve_exact(values.len())
            .map_err(|cause|fail(PreparedBoundaryFrameCause::Capacity(cause)))?;
        for (ordinal,(value,role)) in values.into_iter().zip(actual_roles).enumerate(){
            let layout=Header::new(route,contract.schema(),ordinal,role)
                .map_err(|_|fail(PreparedBoundaryFrameCause::Contract))?;
            funding.reserve_metadata(layout.bytes()).map_err(|cause|fail(PreparedBoundaryFrameCause::Funding(cause)))?;
            let mut header=Vec::new();header.try_reserve_exact(layout.bytes())
                .map_err(|cause|fail(PreparedBoundaryFrameCause::Capacity(cause)))?;
            layout.write(&mut header);frames.push(crate::RoleExactBoundaryValue::new(header,value));
        }
        Ok(PreparedBoundaryFrames{values:frames,route,source:self.clone(),funding:funding.clone()})
    }
}

/// One owned declaration/account loan for a reached route. It carries no native
/// group or model state and does not turn a PP control context into neural TP.
#[derive(Clone,Debug)]
pub struct PreparedBoundarySource {
    route:CommunicationRouteId,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl PreparedBoundarySource {
    pub(crate) fn error(&self,cause:PreparedBoundaryFrameCause)->PreparedBoundaryFrameError{
        PreparedBoundaryFrameError{cause,_source:self.source.clone(),_funding:self.funding.clone()}
    }

    /// Exact declaration source bound by the backend's selected route loan.
    pub fn source(&self)->&RetainedCommunicationSource{&self.source}
    /// Finite destination funding for this boundary attempt.
    pub fn funding(&self)->&WorkspaceMetadataFunding{&self.funding}
    /// Actual selected route descriptor.
    pub fn descriptor(&self)->&CommunicationRouteDescriptor{
        self.source.manifest().routes().iter().find(|candidate|candidate.id()==self.route)
            .expect("closed selected boundary source")
    }
    /// Creates headers from this selected source using the shared writer.
    pub fn frame_values<T>(&self,roles:&[BoundaryRoleContract],values:Vec<T>)
        ->Result<PreparedBoundaryFrames<T>,PreparedBoundaryFrameError>{
        self.source.prepare_boundary_frames(self.route,roles,values,&self.funding)
    }
}
impl RetainedCommunicationSource {
    /// Own a descriptive route/account loan after the backend has authenticated
    /// its actual native resource. This creates no original submission grant.
    pub fn prepare_boundary_source(&self,route:CommunicationRouteId,funding:&WorkspaceMetadataFunding)
        ->Result<PreparedBoundarySource,PreparedBoundaryFrameError>{
        let fail=|cause|PreparedBoundaryFrameError{cause,_source:self.clone(),_funding:funding.clone()};
        let parts=[size_of::<PreparedBoundarySource>(),size_of::<Result<PreparedBoundarySource,PreparedBoundaryFrameError>>(),
            size_of::<PreparedBoundaryFrameError>(),size_of::<(&Self,CommunicationRouteId,&WorkspaceMetadataFunding)>(),
            size_of::<Option<&CommunicationRouteDescriptor>>()];
        let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||fail(PreparedBoundaryFrameCause::Funding(WorkspaceMetadataFundingError::Overflow)))?;
        funding.reserve_metadata(bytes).map_err(|cause|fail(PreparedBoundaryFrameCause::Funding(cause)))?;
        if !self.manifest().routes().iter().any(|candidate|candidate.id()==route&&candidate.boundary_contract().is_some()){
            return Err(fail(PreparedBoundaryFrameCause::Contract));
        }
        Ok(PreparedBoundarySource{route,source:self.clone(),funding:funding.clone()})
    }
}

impl RetainedCommunicationSource {
    /// Exact canonical header length for this retained role and actual scalar
    /// geometry, through the same header layout used by the wire writer. This
    /// allocates no shape or bytes and grants no native transfer authority.
    pub fn boundary_header_length(&self,route:CommunicationRouteId,ordinal:usize,shape:&[i32],
        dtype:&TensorDtype,funding:&WorkspaceMetadataFunding)->Result<usize,PreparedBoundaryFrameError>{
        let failed=|cause|PreparedBoundaryFrameError{cause,_source:self.clone(),_funding:funding.clone()};
        let parts=[Header::source_control_bytes().ok_or_else(||failed(PreparedBoundaryFrameCause::Funding(WorkspaceMetadataFundingError::Overflow)))?,
            size_of::<Result<Header<'_>,CommunicationManifestError>>(),
            size_of::<Result<usize,PreparedBoundaryFrameError>>(),size_of::<PreparedBoundaryFrameError>(),
            size_of::<(&Self,CommunicationRouteId,usize,&[i32],&TensorDtype,&WorkspaceMetadataFunding)>(),
            size_of::<(u64,usize,u32)>(),size_of::<Option<u64>>()];
        let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||failed(PreparedBoundaryFrameCause::Funding(WorkspaceMetadataFundingError::Overflow)))?;
        funding.reserve_metadata(bytes).map_err(|cause|failed(PreparedBoundaryFrameCause::Funding(cause)))?;
        let contract=self.manifest().routes().iter().find(|value|value.id()==route)
            .and_then(CommunicationRouteDescriptor::boundary_contract)
            .ok_or_else(||failed(PreparedBoundaryFrameCause::Contract))?;
        let role=contract.roles().get(ordinal).ok_or_else(||failed(PreparedBoundaryFrameCause::Contract))?;
        Header::actual(route,contract.schema(),ordinal,role,shape,dtype).map(|header|header.bytes())
            .map_err(|_|failed(PreparedBoundaryFrameCause::Contract))
    }
}
