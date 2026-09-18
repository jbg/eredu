//! Once-only scheduled target from the actual globally funded fragment source.
use super::*;

/// This is evidence of the existing global charge, not local native credits.
/// The local native loans remain independently consumable by their real ranks.
#[derive(Debug)]
pub(crate) struct PreparedPartitionAssemblyCharge<'a> {
    source: &'a SharedCapturePlan,
    index: usize,
    phase:CapturePhase,
    prediction:u64,
    charged: CaptureUsage,
    _exclusive: std::marker::PhantomData<&'a mut PreparedPartitionFragmentAllowance>,
}
impl PreparedPartitionAssemblyCharge<'_> {
    pub(crate) fn validate(&self, source: &SharedCapturePlan, index: usize) -> bool {
        self.validate_at(source,index,CapturePhase::Prefill,0)
    }
    pub(crate) fn validate_at(&self,source:&SharedCapturePlan,index:usize,phase:CapturePhase,prediction:u64)->bool {
        self.source.same_storage(source)&&self.index==index&&self.phase==phase&&self.prediction==prediction
    }
    pub(crate) fn charged(&self) -> CaptureUsage { self.charged }
}
impl PreparedPartitionFragmentAllowance {
    /// Reuse exactly the assembler's final record equation before its payload
    /// exists. This does not spend the later assembly worker's child credits.
    pub(crate) fn assembly_record_charge(&self, receipt: &PartitionCaptureReceiptPlan)
        -> Result<CaptureUsage, CaptureError>
    {
        if !self.matches(receipt) { return Err(invalid("assembly charge source differs")); }
        let mut charged = receipt.fragment_assembly_usage()?.ok_or_else(|| invalid("missing fragment assembly source"))?;
        for row in &self.rows {
            let (_, usage, _) = self.fragment_record_charge(receipt, row.producer, row.fragment)
                .ok_or_else(|| invalid("missing original fragment charge"))?;
            charged = charged.checked_add(usage)?;
        }
        Ok(charged)
    }
    pub(crate) fn take_assembly_charge(&mut self, receipt: &PartitionCaptureReceiptPlan)
        -> Result<PreparedPartitionAssemblyCharge<'_>, PartitionCaptureFragmentAllowanceError>
    {
        let source=self.source.clone(); let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureFragmentAllowanceError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let parts=[size_of::<PreparedPartitionAssemblyCharge<'_>>()*2,
            size_of::<Result<PreparedPartitionAssemblyCharge<'_>,PartitionCaptureFragmentAllowanceError>>(),
            size_of::<PartitionCaptureFragmentAllowanceError>(),size_of::<Cause>(),size_of::<CaptureUsage>()*3,
            size_of::<Result<CaptureUsage,CaptureError>>(),size_of::<(&mut Self,&PartitionCaptureReceiptPlan)>(),
            size_of::<std::slice::Iter<'_,Row>>(),size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>()];
        self.metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||error(Cause::Source("assembly charge controls overflow")))?)
            .map_err(|cause|error(cause.into()))?;
        if self.assembly_claimed || !self.matches(receipt) {
            return Err(error(Cause::Source("assembly target is unavailable or spent")));
        }
        self.assembly_claimed=true;
        let charged=self.assembly_record_charge(receipt).map_err(|cause|error(cause.into()))?;
        Ok(PreparedPartitionAssemblyCharge{source:&self.source,index:receipt.context().selection_index,
            phase:receipt.context().phase,prediction:receipt.context().prediction,charged,_exclusive:std::marker::PhantomData})
    }
}
