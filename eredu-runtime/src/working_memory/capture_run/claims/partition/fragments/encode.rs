//! Borrow completed local values through the ordinary canonical wire worker.
use super::*;
use eredu_core::capture::{CapturePayloadWire, CaptureRecordWire};
use crate::capture::partition::{PartitionCaptureBuffer, PartitionCaptureEncodingError};

#[derive(Debug,thiserror::Error)]
enum EncodeCause {
    #[error("contiguous producer source, payload or charge differs")]
    Source,
    #[error(transparent)] Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)] Memory(#[from] WorkingMemoryError),
    #[error(transparent)] Encoding(#[from] PartitionCaptureEncodingError),
    #[error(transparent)] Destination(#[from] eredu_nn::Error),
}
/// The exact writer error retains its original metadata and host/source loans.
#[derive(Debug,thiserror::Error)]
#[error("contiguous capture producer: {cause}")]
pub(crate) struct PartitionFragmentEncodingError {
    #[source] cause:EncodeCause,
    _source:SharedCapturePlan,_custody:CaptureTensorCustody,_metadata:WorkspaceMetadataFunding,
}
impl PreparedPartitionFragmentDestinations {
    /// Lend only this rank's actual completed contiguous fragment, or its real
    /// empty source. The surrounding exchange owns unique-rank submission and
    /// voting; encoding grants no native completion or final assembly evidence.
    pub(crate) fn encode_contiguous_producer(&self,receipt:&PartitionCaptureReceiptPlan,
        empty_dtype:Option<&TensorDtype>,funding:&WorkspaceMetadataFunding)
        ->Result<PartitionCaptureBuffer<u8>,PartitionFragmentEncodingError> {
        let error=|cause|PartitionFragmentEncodingError{cause,_source:self.source.clone(),
            _custody:self.custody.share_scheduled(),_metadata:funding.clone()};
        let parts=[size_of::<(&Self,&PartitionCaptureReceiptPlan,Option<&TensorDtype>,&WorkspaceMetadataFunding)>(),
            CaptureRecordWire::control_bytes(),size_of::<CaptureOutcome>(),size_of::<Option<CaptureRecordWire<'_>>>(),
            size_of::<Option<(TensorDtype,CaptureUsage,CaptureUsage)>>(),size_of::<(TensorDtype,CaptureUsage,CaptureUsage)>(),
            size_of::<Option<&PartitionFragmentValue>>(),size_of::<Option<&CaptureSlicePartition>>(),
            size_of::<(usize,usize,u64)>(),size_of::<std::slice::Iter<'_,u64>>(),
            size_of::<PartitionFragmentEncodingError>(),size_of::<EncodeCause>(),
            size_of::<Result<PartitionCaptureBuffer<u8>,PartitionFragmentEncodingError>>()];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||error(EncodeCause::Source))?).map_err(|e|error(e.into()))?;
        self.custody.validate().map_err(|e|error(e.into()))?;
        if !self.matches(receipt){return Err(error(EncodeCause::Source));}
        let producer=self.allowance.local_rank();
        let projection=receipt.producer(producer).ok_or_else(||error(EncodeCause::Source))?;
        if receipt.routed_producer(producer).is_some(){return self.encode_routed_producer(receipt,empty_dtype,funding);}
        if projection.fragments().len()>1{return Err(error(EncodeCause::Source));}
        let maximum=usize::try_from(receipt.max_record_bytes()).map_err(|_|error(EncodeCause::Source))?;
        if projection.fragments().is_empty() {
            if !matches!(empty_dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16)) {
                return Err(error(EncodeCause::Source));
            }
            return crate::capture::partition::encode_contiguous(receipt.context(),receipt.identity(),producer,
                receipt.combination(),empty_dtype,None,maximum,funding).map_err(|e|error(e.into()));
        }
        // Source-derived full record charge includes its ordinary metadata body;
        // the separately spent native quota is never charged a second time.
        let (dtype,charged,_)=self.allowance.fragment_record_charge(receipt,producer,0)
            .ok_or_else(||error(EncodeCause::Source))?;
        let value=self.value(producer,0).ok_or_else(||error(EncodeCause::Source))?;
        value.identity().custody.validate().map_err(|e|error(e.into()))?;
        let index=receipt.context().selection_index;
        let selection=&self.source.admission().plan().selections[index];
        let point=&self.source.admission().points()[index];
        let selected=&projection.fragments()[0].local().shape;
        let available=selected.iter().try_fold(1u64,|n,&v|n.checked_mul(v)).ok_or_else(||error(EncodeCause::Source))?;
        let outcome=crate::capture::completed_capture_outcome(&selection.transform,available);
        let payload=match value {
            PartitionFragmentValue::Tensor(v)=>CapturePayloadWire::Tensor(v.observation().as_observation()),
            PartitionFragmentValue::Summary(v)=>CapturePayloadWire::Summary(v.observation()),
            PartitionFragmentValue::Histogram(v)=>CapturePayloadWire::Histogram(v.observation()),
            PartitionFragmentValue::Routed(_)=>return Err(error(EncodeCause::Source)),
        };
        let record=CaptureRecordWire{schema_version:CAPTURE_SCHEMA_VERSION,selection_id:&selection.id,
            path:&selection.path,node_id:&point.node_id,position:point.position,
            source_shape:Some(projection.local_shape()),source_dtype:Some(&dtype),selected_shape:Some(selected),
            outcome:&outcome,payload:Some(payload),charged};
        crate::capture::partition::encode_contiguous(receipt.context(),receipt.identity(),producer,receipt.combination(),
            Some(&dtype),Some(&record),maximum,funding).map_err(|e|error(e.into()))
    }
}

impl PreparedPartitionFragmentDestinations {
    fn encode_routed_producer(&self,receipt:&PartitionCaptureReceiptPlan,empty_dtype:Option<&TensorDtype>,
        funding:&WorkspaceMetadataFunding)->Result<PartitionCaptureBuffer<u8>,PartitionFragmentEncodingError> {
        let error=|cause|PartitionFragmentEncodingError{cause,_source:self.source.clone(),
            _custody:self.custody.share_scheduled(),_metadata:funding.clone()};
        let parts=[size_of::<(&Self,&PartitionCaptureReceiptPlan,Option<&TensorDtype>,&WorkspaceMetadataFunding)>(),
            size_of::<Vec<CaptureRecordWire<'_>>>(),CaptureRecordWire::control_bytes(),
            size_of::<Option<&TensorDtype>>(),size_of::<Option<(TensorDtype,CaptureUsage,CaptureUsage)>>(),
            size_of::<(TensorDtype,CaptureUsage,CaptureUsage)>(),size_of::<CaptureOutcome>(),
            size_of::<Option<&ClaimedPartitionRoutedUnits>>(),size_of::<PartitionFragmentEncodingError>(),
            size_of::<Result<Vec<CaptureRecordWire<'_>>,eredu_nn::Error>>(),
            size_of::<Result<PartitionCaptureBuffer<u8>,PartitionFragmentEncodingError>>()];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||error(EncodeCause::Source))?).map_err(|e|error(e.into()))?;
        self.custody.validate().map_err(|e|error(e.into()))?;
        let producer=self.allowance.local_rank();
        if !self.matches(receipt)||receipt.routed_producer(producer).is_none(){return Err(error(EncodeCause::Source));}
        let projection=receipt.producer(producer).ok_or_else(||error(EncodeCause::Source))?;
        // An actual idle source keeps absent result precision on the wire;
        // the common allowance precision only funds its fixed empty Host owner.
        let dtype=empty_dtype;
        if !matches!(dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16|TensorDtype::F64)) {
            return Err(error(EncodeCause::Source));
        }
        let index=receipt.context().selection_index;
        let selection=&self.source.admission().plan().selections[index];let point=&self.source.admission().points()[index];
        let outcome=CaptureOutcome::Captured;
        let mut records=funding.metadata_vec(projection.fragments().len()).map_err(|e|error(e.into()))?;
        for (fragment,geometry) in projection.fragments().iter().enumerate() {
            let (precision,charged,_)=self.allowance.fragment_record_charge(receipt,producer,fragment)
                .ok_or_else(||error(EncodeCause::Source))?;
            if dtype.is_some_and(|actual|actual!=&precision){return Err(error(EncodeCause::Source));}
            let value=self.routed_value(producer,fragment).ok_or_else(||error(EncodeCause::Source))?;
            if dtype.is_none()&&(value.native_rows()!=0||!value.observation().rows.is_empty()||!value.observation().source_token_ranges.is_empty()) {
                return Err(error(EncodeCause::Source));
            }
            value.identity().custody.validate().map_err(|e|error(e.into()))?;
            records.push(CaptureRecordWire{schema_version:CAPTURE_SCHEMA_VERSION,selection_id:&selection.id,
                path:&selection.path,node_id:&point.node_id,position:point.position,
                source_shape:Some(projection.local_shape()),source_dtype:dtype,selected_shape:Some(&geometry.local().shape),
                outcome:&outcome,payload:Some(CapturePayloadWire::RoutedUnits(value.observation())),charged});
        }
        crate::capture::partition::encode_fragment_records(receipt.context(),receipt.identity(),producer,
            receipt.combination(),dtype,&records,usize::try_from(receipt.max_record_bytes()).map_err(|_|error(EncodeCause::Source))?,funding)
            .map_err(|e|error(e.into()))
    }
}
