//! Same prospective receipt costs with precision held behind the source vote.
use super::*;

pub(super) enum Sources<'s,'v> {
    Typed(&'s [PartitionCaptureFragmentSource<'v>]),
    Geometry(&'s [PartitionCaptureFragmentGeometry<'v>]),
    RoutedTyped(&'s [PartitionCaptureRoutedFragmentSource<'v>]),
    RoutedGeometry(&'s [PartitionCaptureRoutedFragmentGeometry<'v>]),
}
pub(super) enum Geometry<'a> {
    Tensor{shape:&'a [u64],slice:&'a ResolvedCaptureSlice,transform:&'a CaptureTransform},
    Routed(PartitionRoutedUnitCaptureRequest<'a>),
}
pub(super) struct Source<'a> {
    pub(super) producer:usize,pub(super) fragment:usize,pub(super) geometry:Geometry<'a>,
    pub(super) dtype:Option<&'a TensorDtype>,pub(super) estimate:PartitionCaptureNativeEstimate,
}
impl Sources<'_,'_> {
    pub(super) fn routed(&self)->bool {matches!(self,Self::RoutedTyped(_)|Self::RoutedGeometry(_))}
    pub(super) fn len(&self)->usize {match self {Self::Typed(rows)=>rows.len(),Self::Geometry(rows)=>rows.len(),
        Self::RoutedTyped(rows)=>rows.len(),Self::RoutedGeometry(rows)=>rows.len()}}
    pub(super) fn get(&self,index:usize)->Option<Source<'_>> {match self {
        Self::Typed(rows)=>rows.get(index).map(|row|Source{producer:row.producer,fragment:row.fragment,
            geometry:Geometry::Tensor{shape:row.local_shape,slice:row.local_slice,transform:row.transform},dtype:Some(&row.dtype),estimate:row.estimate}),
        Self::Geometry(rows)=>rows.get(index).map(|row|Source{producer:row.producer,fragment:row.fragment,
            geometry:Geometry::Tensor{shape:row.local_shape,slice:row.local_slice,transform:row.transform},dtype:None,estimate:row.estimate}),
        Self::RoutedTyped(rows)=>rows.get(index).map(|row|Source{producer:row.geometry.producer,fragment:row.geometry.fragment,
            geometry:Geometry::Routed(row.geometry.request),dtype:Some(&row.dtype),estimate:row.geometry.estimate}),
        Self::RoutedGeometry(rows)=>rows.get(index).map(|row|Source{producer:row.producer,fragment:row.fragment,
            geometry:Geometry::Routed(row.request),dtype:None,estimate:row.estimate}),
    }}
}

/// Original quota with no callable native-source or destination interface.
/// Only the shared coordinated row can consume it after its Source vote.
#[derive(Debug)]
pub(crate) struct PreparedPartitionFragmentSourceAllowance {
    allowance:PreparedPartitionFragmentAllowance,
}
#[derive(Debug,thiserror::Error)]
#[error("partition fragment source vote binding differs")]
pub(crate) struct SourceBindingError {
    _owner:PreparedPartitionFragmentSourceAllowance,
}
impl PreparedPartitionFragmentSourceAllowance {
    pub(crate) fn prepare<T:PartitionCaptureTransport>(transport:&T,receipt:&mut PartitionCaptureReceiptPlan,
        geometry:&[PartitionCaptureFragmentGeometry<'_>],metadata:&WorkspaceMetadataFunding,ledger:&mut dyn CaptureReservation)
        ->Result<Self,PartitionCaptureFragmentAllowanceError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        metadata.reserve_metadata(Self::control_bytes().ok_or_else(||PartitionCaptureFragmentAllowanceError{
            cause:Cause::Source("source allowance controls overflow"),_source:receipt.shared_plan_source().cloned(),_metadata:metadata.clone()})?)
            .map_err(|e|PartitionCaptureFragmentAllowanceError{cause:e.into(),_source:receipt.shared_plan_source().cloned(),_metadata:metadata.clone()})?;
        let allowance=PreparedPartitionFragmentAllowance::prepare_sources(transport,receipt,Sources::Geometry(geometry),metadata,ledger)?;
        Ok(Self{allowance})
    }
    pub(crate) fn prepare_routed<T:PartitionCaptureTransport>(transport:&T,receipt:&mut PartitionCaptureReceiptPlan,
        geometry:&[PartitionCaptureRoutedFragmentGeometry<'_>],metadata:&WorkspaceMetadataFunding,ledger:&mut dyn CaptureReservation)
        ->Result<Self,PartitionCaptureFragmentAllowanceError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        metadata.reserve_metadata(Self::control_bytes().ok_or_else(||PartitionCaptureFragmentAllowanceError{
            cause:Cause::Source("source allowance controls overflow"),_source:receipt.shared_plan_source().cloned(),_metadata:metadata.clone()})?)
            .map_err(|e|PartitionCaptureFragmentAllowanceError{cause:e.into(),_source:receipt.shared_plan_source().cloned(),_metadata:metadata.clone()})?;
        let allowance=PreparedPartitionFragmentAllowance::prepare_sources(transport,receipt,Sources::RoutedGeometry(geometry),metadata,ledger)?;
        Ok(Self{allowance})
    }
    /// Binding changes neither the original global ledger nor any child quota.
    /// The caller is the same source-vote consumer as complete rows.
    pub(crate) fn bind(mut self,receipt:&PartitionCaptureReceiptPlan,dtype:TensorDtype)
        ->Result<PreparedPartitionFragmentAllowance,SourceBindingError> {
        if !(matches!(dtype,TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16)
            || dtype==TensorDtype::F64 && receipt.producers().any(|(rank,_)|receipt.routed_producer(rank).is_some()))
            ||!self.allowance.matches(receipt)||self.allowance.rows.iter().any(|row|row.taken||row.dtype.is_some()) {
            return Err(SourceBindingError{_owner:self});
        }
        for row in &mut self.allowance.rows {row.dtype=Some(dtype.clone());}
        Ok(self.allowance)
    }
    pub(crate) fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<SourceBindingError>(),size_of::<PreparedPartitionFragmentAllowance>()*2,
            size_of::<Result<Self,PartitionCaptureFragmentAllowanceError>>(),
            size_of::<Result<PreparedPartitionFragmentAllowance,SourceBindingError>>(),
            size_of::<(Self,&PartitionCaptureReceiptPlan,TensorDtype)>(),size_of::<Sources<'_,'_>>(),size_of::<Source<'_>>(),
            size_of::<Option<TensorDtype>>(),size_of::<std::slice::IterMut<'_,Row>>(),size_of::<std::slice::Iter<'_,Row>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<SourceBindingError>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
