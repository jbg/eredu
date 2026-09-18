//! Original-account host preparation before a forward receipt is bound.
use super::*;
use crate::capture::partition::PartitionCaptureReceiptLimits;

/// Exact original H with separate identities for every destination. It contains
/// no native scope or transferable allocator. The fixed table is allocated by
/// the same worker used by immediate fragment preparation.
#[derive(Debug)]
pub(super) struct FragmentHostStorage {
    pub(super) rows:Vec<Slot>,pub(super) routed_merge:Vec<bool>,pub(super) scratch:Option<(usize,usize,CaptureTensorCustody)>,
    pub(super) prefill:Option<eredu_core::InferenceGeometry>,pub(super) custody:CaptureTensorCustody,
}
impl FragmentHostStorage {
    pub(super) fn error(&self,source:&SharedCapturePlan,cause:FragmentCause,value:Option<PartitionFragmentValue>)
        ->PartitionFragmentDestinationError {
        PartitionFragmentDestinationError{cause,_value:value,_source:source.clone(),_custody:Some(self.custody.share_scheduled())}
    }
    pub(super) fn bind(self,receipt:&PartitionCaptureReceiptPlan,allowance:PreparedPartitionFragmentAllowance)
        ->PreparedPartitionFragmentDestinations {
        let mut identity=[0;64];identity.copy_from_slice(receipt.identity().as_bytes());
        PreparedPartitionFragmentDestinations{rows:self.rows,routed_merge:self.routed_merge,identity,allowance,
            source:receipt.shared_plan_source().clone(),scratch:self.scratch,
            prefill:self.prefill,custody:self.custody}
    }
}
/// Move-only original Host destinations retained until the real forward receipt
/// exists. The prototype supplies geometry only; it is never a submitted receipt.
#[derive(Debug)]
pub struct PreparedPartitionFragmentHostFunding {
    prototype:PartitionCaptureReceiptPlan,storage:FragmentHostStorage,bytes:u64,
}
/// A refused bind retains all original per-fragment H and the actual allowance.
/// No receipt, slot or native attempt can be recovered through this error.
#[derive(Debug,thiserror::Error)]
#[error("prepared partition fragment Host binding: {cause}")]
pub struct PartitionFragmentHostBindingError {
    #[source] cause:CaptureRunHostError,
    _allowance:PreparedPartitionFragmentAllowance,_owner:PreparedPartitionFragmentHostFunding,
}
#[derive(Debug,thiserror::Error)]
enum PreparationCause {
    #[error(transparent)] Source(#[from] CaptureRunHostError),
    #[error(transparent)] Destination(#[from] PartitionFragmentDestinationError),
}
/// Preparation refusal preserves its supplied source and any reached Host hold.
#[derive(Debug,thiserror::Error)]
#[error("prepared partition fragment Host source: {cause}")]
pub struct PartitionFragmentHostPreparationError {
    #[source] cause:PreparationCause,_prototype:PartitionCaptureReceiptPlan,
}
impl WorkingMemoryFundingRun {
    /// Protect actual fragment/table destinations from this original reservation
    /// before its funding run is lent to the enclosing generation driver. The
    /// later receipt still must match every source/shape/projection and limit.
    pub fn prepare_partition_fragment_host(&self,reservation:&WorkingMemoryReservation,
        prototype:PartitionCaptureReceiptPlan,prefill:Option<eredu_core::InferenceGeometry>)
        ->Result<PreparedPartitionFragmentHostFunding,PartitionFragmentHostPreparationError> {
        let result=(||->Result<(FragmentHostStorage,u64),PreparationCause> {
            let plan=match prefill {
                Some(geometry)=>PartitionFragmentHostPlan::prepare_prefill(&prototype,geometry),
                None=>PartitionFragmentHostPlan::prepare(&prototype),
            }?;
            let bytes=plan.initialization_peak_bytes();
            let storage=self.prepare_fragment_host_storage(reservation,&plan)?;
            Ok((storage,bytes))
        })();
        match result {
            Ok((storage,bytes))=>Ok(PreparedPartitionFragmentHostFunding{prototype,storage,bytes}),
            Err(cause)=>Err(PartitionFragmentHostPreparationError{cause,_prototype:prototype}),
        }
    }
}
impl PreparedPartitionFragmentHostFunding {
    /// Exact limits used by this retained per-selection Host source. A broader
    /// program envelope does not replace the source's original receipt limits.
    pub fn original_receipt_limits(&self)->PartitionCaptureReceiptLimits {
        self.prototype.original_limits()
    }
    /// Descriptive protected capacity, independent from cumulative capture usage.
    pub const fn protected_bytes(&self)->u64 {self.bytes}
    /// Consume this owner once. The actual source/epoch receipt and native
    /// allowance are mandatory; equal capacity alone cannot bind destinations.
    pub fn bind(self,receipt:&PartitionCaptureReceiptPlan,allowance:PreparedPartitionFragmentAllowance)
        ->Result<PreparedPartitionFragmentDestinations,PartitionFragmentHostBindingError> {
        let result=(|| {
            self.storage.custody.validate()?;
            if !self.prototype.same_fragment_host_source(receipt)||!allowance.matches(receipt) {
                return Err(CaptureRunHostError::ReceiptMismatch);
            }
            let plan=match self.storage.prefill {
                Some(geometry)=>PartitionFragmentHostPlan::prepare_prefill(receipt,geometry)?,
                None=>PartitionFragmentHostPlan::prepare(receipt)?,
            };
            if plan.initialization_peak_bytes()!=self.bytes||plan.slots!=self.storage.rows.len() {
                return Err(CaptureRunHostError::ReceiptMismatch);
            }
            for slot in &self.storage.rows {
                slot.custody.validate()?;
                if allowance.fragment_source(slot.producer,slot.fragment).is_none(){return Err(CaptureRunHostError::ReceiptMismatch);}
            }
            if let Some((_,_,custody))=&self.storage.scratch {custody.validate()?;}
            Ok(())
        })();
        match result {
            Ok(())=>Ok(self.storage.bind(receipt,allowance)),
            Err(cause)=>Err(PartitionFragmentHostBindingError{cause,_allowance:allowance,_owner:self}),
        }
    }
}
pub(super) fn control_bytes()->Option<usize> {
    let parts=[size_of::<FragmentHostStorage>()*2,size_of::<PreparedPartitionFragmentHostFunding>()*2,
        size_of::<PartitionFragmentHostBindingError>(),size_of::<PartitionCaptureReceiptPlan>(),
        size_of::<Result<PreparedPartitionFragmentHostFunding,PartitionFragmentHostPreparationError>>(),
        size_of::<PartitionFragmentHostPreparationError>(),size_of::<PreparationCause>(),
        size_of::<Result<(FragmentHostStorage,u64),PreparationCause>>(),
        size_of::<Result<PreparedPartitionFragmentDestinations,PartitionFragmentHostBindingError>>(),
        size_of::<Result<FragmentHostStorage,PartitionFragmentDestinationError>>(),
        size_of::<(&WorkingMemoryFundingRun,&WorkingMemoryReservation,&PartitionFragmentHostPlan<'_>)>(),
        size_of::<(PreparedPartitionFragmentHostFunding,&PartitionCaptureReceiptPlan,PreparedPartitionFragmentAllowance)>(),
        size_of::<Option<eredu_core::InferenceGeometry>>(),size_of::<Result<(),CaptureRunHostError>>(),
        size_of::<std::slice::Iter<'_,Slot>>(),size_of::<Option<&CaptureTensorCustody>>(),
        size_of::<[u8;64]>(),size_of::<u64>(),size_of::<usize>()*3,
        size_of::<(&PartitionCaptureReceiptPlan,&PartitionCaptureReceiptPlan)>(),size_of::<bool>()*4,
        size_of::<Option<(&SharedCapturePlan,&SharedCapturePlan)>>(),size_of::<(&PartitionCaptureContext,&PartitionCaptureContext)>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
