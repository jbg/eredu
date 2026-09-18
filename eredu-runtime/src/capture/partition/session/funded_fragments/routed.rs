//! Actual sparse request source joined to the existing fragment allowance.
use super::*;
/// Exact sparse source geometry before its scalar vote. Receive-order rows,
/// logical source tokens, expert ownership and global unit coordinates remain
/// separate; none is reinterpreted as a dense local tensor.
#[derive(Debug,Clone,Copy)]
pub struct PartitionCaptureRoutedFragmentGeometry<'a> {
    /// Authoritative producer rank.
    pub producer:usize,
    /// Receipt fragment in this producer's exact local traversal order.
    pub fragment:usize,
    /// Same request consumed by the ordinary partition sparse estimator.
    pub request:PartitionRoutedUnitCaptureRequest<'a>,
    /// Actual selected native collector estimate for this immutable request.
    pub estimate:PartitionCaptureNativeEstimate,
}
/// An exact sparse source with its actual physical scalar witness.
#[derive(Debug)]
pub struct PartitionCaptureRoutedFragmentSource<'a> {
    /// Original request, selected ownership and native estimate.
    pub geometry:PartitionCaptureRoutedFragmentGeometry<'a>,
    /// Actual native source scalar; never inferred from the logical layout.
    pub dtype:TensorDtype,
}
impl PreparedPartitionFragmentAllowance {
    /// The ordinary global/local receipt cost equation with exact sparse inputs.
    /// Native loans remain unavailable until this typed source is coordinated.
    pub fn prepare_routed<T:PartitionCaptureTransport>(transport:&T,receipt:&mut PartitionCaptureReceiptPlan,
        sources:&[PartitionCaptureRoutedFragmentSource<'_>],metadata:&HostMetadataFunding,
        ledger:&mut dyn CaptureReservation)->Result<Self,PartitionCaptureFragmentAllowanceError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        Self::prepare_sources(transport,receipt,Sources::RoutedTyped(sources),metadata,ledger)
    }
}
pub(super) fn matches(receipt:&PartitionCaptureReceiptPlan,rank:usize,fragment:usize,
    request:PartitionRoutedUnitCaptureRequest<'_>)->bool
{
    let Some(projection)=receipt.producer(rank) else{return false};
    let Some(ownership)=receipt.routed_producer(rank) else{return false};
    let Ok(bank)=super::super::routed::geometry(receipt) else{return false};
    let Ok(axes)=eredu_core::capture::CaptureRoutedUnitsGeometry::fragment_axes(projection,fragment) else{return false};
    request.geometry==bank && request.source_tokens==projection.global_shape()[0]
        && request.ownership==ownership && request.validate().is_ok()
        && request.slice.starts.as_slice()==axes[0] && request.slice.ends.as_slice()==axes[1]
        && request.slice.strides.as_slice()==axes[2] && request.slice.shape.as_slice()==axes[3]
}
pub(super) fn control_bytes()->Option<usize> {
    let parts=[size_of::<PartitionCaptureRoutedFragmentGeometry<'_>>() * 2,
        size_of::<PartitionCaptureRoutedFragmentSource<'_>>(),size_of::<PartitionRoutedUnitCaptureRequest<'_>>() * 2,
        size_of::<RoutedUnitGeometry>(),size_of::<[[u64;3];4]>(),size_of::<[u64;3]>(),
        size_of::<Result<[[u64;3];4],eredu_core::capture::RoutedUnitValidationError>>(),
        size_of::<Result<RoutedUnitGeometry,CaptureError>>(),size_of::<Option<&RoutedUnitCaptureOwnership>>(),
        size_of::<(&PartitionCaptureReceiptPlan,usize,usize,PartitionRoutedUnitCaptureRequest<'_>)>(),
        eredu_core::capture::CaptureRoutedUnitsGeometry::partition_control_bytes()?];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
