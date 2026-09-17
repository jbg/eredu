//! Closed scheduled-frame consumer of the existing all-rank receipt protocol.
use super::*;
use crate::capture::partition::{PartitionCaptureDecoder, PartitionCaptureExchange,
    PartitionCaptureExchangeError, PartitionCapturePayload, PartitionCaptureReceiptPlan,
    PartitionCaptureRecordEncoding, PartitionCaptureTransport};
use eredu_core::{Completion, capture::PartitionCaptureCombination};
#[derive(Clone, Copy)]
enum PayloadKind { Tensor, Summary, Histogram, Candidates, Scores }


#[derive(Debug,thiserror::Error)]
enum DeliveryCause {
    #[error("scheduled partition source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Exchange(#[from] PartitionCaptureExchangeError),
}
/// Receipt delivery failure retains both the immutable admission and original
/// frame H. The same native error remains a source; no rejection refunds work.
#[derive(Debug,thiserror::Error)]
#[error("scheduled partition tensor delivery: {cause}")]
pub struct PartitionCaptureTensorDeliveryError {
    #[source]
    cause:DeliveryCause,
    _source:Option<SharedCapturePlan>,
    _custody:Option<CaptureTensorCustody>,
    _metadata:WorkspaceMetadataFunding,
}

/// One prepared complete-global tensor delivery through the ordinary protocol.
/// Construction authenticates a retained receipt and pays the closed decoder's
/// controls before forward. The receipt's placement and native transport source
/// remain separately admitted; this is not a public native allocation grant.
#[must_use="deliver once through the selected all-rank post-forward boundary"]
pub struct PreparedPartitionTensorDelivery<'t,T:PartitionCaptureTransport> {
    evidence:Option<crate::capture::partition::PreparedPartitionCaptureEvidence>,
    exchange:PartitionCaptureExchange<'t,T>,
    source:SharedCapturePlan,
    producer:usize,
    dtype:Option<TensorDtype>,
    kind:PayloadKind,
    charged:CaptureUsage,
    metadata:WorkspaceMetadataFunding,
}
impl<T:PartitionCaptureTransport> fmt::Debug for PreparedPartitionTensorDelivery<'_,T> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("PreparedPartitionTensorDelivery").field("producer",&self.producer).finish_non_exhaustive()
    }
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionTensorDelivery<'t,T>
where T::Error:Send+Sync+'static, <T::Completion as Completion>::Error:Send+Sync+'static {
    /// Prepare only an actual single complete producer with the exact shared
    /// plan. Call on all ranks before the enclosing forward admission agreement.
    pub fn prepare(exchange:PartitionCaptureExchange<'t,T>, dtype:TensorDtype,
        charged:CaptureUsage, metadata:&WorkspaceMetadataFunding)
        -> Result<Self,PartitionCaptureTensorDeliveryError> {
        Self::prepare_with_source(exchange, Some(dtype), charged, metadata)
    }

    pub(crate) fn prepare_with_source(exchange:PartitionCaptureExchange<'t,T>, dtype:Option<TensorDtype>,
        charged:CaptureUsage, metadata:&WorkspaceMetadataFunding)
        -> Result<Self,PartitionCaptureTensorDeliveryError> {
        let receipt=exchange.receipt_plan();
        let source=receipt.shared_plan_source().cloned();
        let error=|cause|PartitionCaptureTensorDeliveryError { cause,
            _source:source.clone(), _custody:None, _metadata:metadata.clone() };
        let controls=[size_of::<Self>(),size_of::<PartitionCaptureTensorDeliveryError>(),
            size_of::<DeliveryCause>(),size_of::<Result<Self,PartitionCaptureTensorDeliveryError>>(),
            size_of::<Result<(),PartitionCaptureTensorDeliveryError>>(),
            size_of::<Option<CaptureTensorCustody>>(),size_of::<Option<SharedCapturePlan>>(),
            size_of::<(&PartitionCaptureReceiptPlan,usize,bool)>(),
            size_of::<CaptureTensorGeometry<'_>>(), size_of::<CaptureSummaryGeometry<'_>>(),
            size_of::<PayloadKind>() * 2,
            size_of::<Option<CaptureTensorGeometry<'_>>>(),
            size_of::<CaptureHistogramGeometry<'_>>(),
            size_of::<Option<CaptureHistogramGeometry<'_>>>(),
            size_of::<Option<crate::capture::partition::CompleteVocabularyGeometry<'_>>>()*2,
            size_of::<Result<Option<crate::capture::partition::CompleteVocabularyGeometry<'_>>,CaptureTensorGeometryError>>(),
            size_of::<Option<usize>>(),
            size_of::<Option<CaptureSummaryGeometry<'_>>>(),
            size_of::<(&[usize], bool)>(),size_of::<Option<(usize,&CaptureSlicePartition)>>(),
            size_of::<CaptureUsage>()*2,size_of::<Option<CaptureUsage>>(),
            // Closed for_each worker: one decoder/receipt loan and received flag.
            size_of::<(&mut TensorDecoder<'_, '_>,&PartitionCaptureReceiptPlan,bool)>(),
            PartitionCaptureExchange::<T>::decoder_control_bytes::<TensorDecoder<'_, '_>>()
                .ok_or_else(||error(DeliveryCause::Source("exchange controls overflow")))?];
        let bytes=controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or_else(||error(DeliveryCause::Source("delivery controls overflow")))?;
        metadata.reserve_metadata(bytes).map_err(|cause|error(cause.into()))?;
        let admitted=source.as_ref().ok_or_else(||error(DeliveryCause::Source("receipt has no original shared admission")))?;
        let mut producers=receipt.producers();
        let (producer,projection)=producers.next().ok_or_else(||error(DeliveryCause::Source("receipt has no producer")))?;
        if producers.next().is_some() || receipt.combination()!=PartitionCaptureCombination::Disjoint
            || projection.fragments().len()!=1 || projection.local_shape()!=projection.global_shape()
            || receipt.context().invocation.is_some() || receipt.routed_producer(producer).is_some()
            || !matches!(dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16))
        {return Err(error(DeliveryCause::Source("receipt is not one complete floating tensor producer")));}
        let index = receipt.context().selection_index;
        metadata.reserve_metadata(super::vocabulary::exchange_control_bytes(
            &admitted.admission().plan().selections[index].transform)
            .ok_or_else(||error(DeliveryCause::Source("vocabulary exchange controls overflow")))?)
            .map_err(|cause|error(cause.into()))?;
        let summary = matches!(admitted.admission().plan().selections[index].transform, CaptureTransform::Summary);
        let histogram=matches!(admitted.admission().plan().selections[index].transform,CaptureTransform::Histogram { .. });
        let vocabulary=matches!(admitted.admission().plan().selections[index].transform,CaptureTransform::TopCandidates { .. }|CaptureTransform::TokenScores { .. });
        let controls=crate::capture::partition::CompleteVocabularyGeometry::control_bytes()
            .ok_or_else(||error(DeliveryCause::Source("terminal geometry controls overflow")))?;
        metadata.reserve_metadata(controls).map_err(|cause|error(cause.into()))?;
        let vocabulary_geometry=crate::capture::partition::CompleteVocabularyGeometry::prepare(admitted.admission(),index,
            receipt.context().phase,receipt.context().prediction,
            if vocabulary {projection.global_shape().get(1).and_then(|n|usize::try_from(*n).ok())} else {None})
            .map_err(|_|error(DeliveryCause::Source("receipt terminal vocabulary source differs")))?;
        let tensor_geometry = if summary || histogram || vocabulary { None } else {
            Some(CaptureTensorGeometry::prepare(admitted.admission(), index,
                receipt.context().phase, receipt.context().prediction, None)
                .map_err(|_|error(DeliveryCause::Source("receipt selection has no scheduled tensor destination")))?)
        };
        let summary_geometry = if summary {
            Some(CaptureSummaryGeometry::prepare(admitted.admission(), index,
                receipt.context().phase, receipt.context().prediction, None)
                .map_err(|_|error(DeliveryCause::Source("receipt selection has no scheduled summary destination")))?)
        } else { None };
        let histogram_geometry=if histogram {
            let controls=CaptureHistogramGeometry::preparation_control_bytes()
                .ok_or_else(||error(DeliveryCause::Source("histogram geometry controls overflow")))?;
            metadata.reserve_metadata(controls).map_err(|cause|error(cause.into()))?;
            Some(CaptureHistogramGeometry::prepare(admitted.admission(),index,
                receipt.context().phase,receipt.context().prediction,None)
                .map_err(|_|error(DeliveryCause::Source("receipt selection has no scheduled histogram destination")))?)
        } else {None};
        let shape = match (&tensor_geometry, &summary_geometry, &histogram_geometry, &vocabulary_geometry) {
            (Some(geometry), None, None, None) => geometry.source_shape(),
            (None, Some(geometry), None, None) => geometry.source_shape(),
            (None,None,Some(geometry),None) => geometry.source_shape(),
            (None,None,None,Some(geometry))=>&geometry.shape()[..],
            _ => return Err(error(DeliveryCause::Source("receipt payload selection differs"))),
        };
        if shape.len()!=projection.global_shape().len()
            || !shape.iter().zip(projection.global_shape()).all(|(a,b)|u64::try_from(*a).ok()==Some(*b))
        {return Err(error(DeliveryCause::Source("receipt source geometry differs from scheduled admission")));}
        let kind = if summary { PayloadKind::Summary } else if histogram {PayloadKind::Histogram}
            else if matches!(admitted.admission().plan().selections[index].transform,CaptureTransform::TopCandidates{..}){PayloadKind::Candidates}
            else if vocabulary {PayloadKind::Scores}else { PayloadKind::Tensor };
        drop(producers);
        drop((tensor_geometry, summary_geometry, histogram_geometry, vocabulary_geometry));
        let source=source.expect("validated original source");
        Ok(Self {evidence:None,exchange,source,producer,dtype,kind,charged,metadata:metadata.clone()})
    }

    pub(crate) fn receipt_plan(&self) -> &PartitionCaptureReceiptPlan { self.exchange.receipt_plan() }

    pub(crate) fn bind_source_dtype(&mut self, dtype: TensorDtype) -> Result<(), PartitionCaptureTensorDeliveryError> {
        if !matches!(dtype, TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
            || self.dtype.as_ref().is_some_and(|current| current != &dtype) {
            return Err(PartitionCaptureTensorDeliveryError { cause:DeliveryCause::Source("resolved producer scalar differs"),
                _source:Some(self.source.clone()),_custody:None,_metadata:self.metadata.clone() });
        }
        self.dtype = Some(dtype);
        Ok(())
    }

    pub(crate) fn with_evidence(mut self, evidence:crate::capture::partition::PreparedPartitionCaptureEvidence)->Self {
        self.evidence=Some(evidence);self
    }

    /// Complete this prepared receipt at the shared all-rank post-forward point.
    /// A producer borrows its finished original record. A receiver fills its own
    /// original claim; final delivery voting precedes the enclosing frame seal.
    pub fn deliver(self, frame:&mut ScheduledCaptureStep<'_>) -> Result<(),PartitionCaptureTensorDeliveryError> {
        let custody=frame.claim.custody.share_scheduled();
        let dtype = match self.dtype.clone() {
            Some(dtype) => dtype,
            None => return Err(PartitionCaptureTensorDeliveryError {
                cause:DeliveryCause::Source("producer scalar was not resolved before delivery"),
                _source:Some(self.source.clone()),_custody:Some(custody),_metadata:self.metadata.clone(),
            }),
        };
        let receipt=self.exchange.receipt_plan();
        let context=receipt.context();
        let index=context.selection_index;
        let rank=self.exchange.local_rank();
        let local=(|| {
            if !std::ptr::eq(frame.claim.source.admission(),self.source.admission())
                || frame.claim.phase!=context.phase || frame.claim.prediction!=context.prediction
                || frame.claim.invocation.is_some() || frame.claim.window.is_some()
            {return Err(PartitionCaptureExchangeError::Protocol("scheduled capture source differs"));}
            frame.claim.custody.validate().map_err(CaptureRunHostError::from)?;
            if rank!=self.producer {return Ok(None);}
            let record=frame.records().get(index).ok_or(PartitionCaptureExchangeError::Protocol("scheduled producer record missing"))?;
            if !matches!(record.outcome,CaptureOutcome::Captured|CaptureOutcome::Truncated{..})
                || !matches!((self.kind, &record.payload),
                    (PayloadKind::Tensor, Some(CapturePayload::SharedTensor(_)))
                    | (PayloadKind::Summary, Some(CapturePayload::Summary(_)))
                    | (PayloadKind::Histogram, Some(CapturePayload::Histogram(_)))
                    | (PayloadKind::Candidates,Some(CapturePayload::Candidates(_)))
                    | (PayloadKind::Scores,Some(CapturePayload::TokenScores(_))))
                || record.source_dtype.as_ref()!=Some(&dtype) || record.charged!=self.charged
            {return Err(PartitionCaptureExchangeError::Protocol("scheduled producer record differs"));}
            let maximum=usize::try_from(receipt.max_record_bytes()).map_err(|_|CaptureError::Overflow)?;
            Ok(Some(PartitionCaptureRecordEncoding::new(context,receipt.identity(),rank,
                receipt.combination(),record,maximum).encode(&self.metadata)?))
        })();
        let decoder=TensorDecoder {frame,rank,producer:self.producer,dtype,
            kind:self.kind,charged:self.charged,metadata:&self.metadata,evidence:self.evidence};
        self.exchange.exchange_into(local,decoder).map_err(|cause|PartitionCaptureTensorDeliveryError {
            cause:cause.into(),_source:Some(self.source),_custody:Some(custody),_metadata:self.metadata,
        })
    }
}

struct TensorDecoder<'f,'a> {
    frame:&'f mut ScheduledCaptureStep<'a>,rank:usize,producer:usize,dtype:TensorDtype,
    kind:PayloadKind,charged:CaptureUsage,metadata:&'f WorkspaceMetadataFunding,
    evidence:Option<crate::capture::partition::PreparedPartitionCaptureEvidence>,
}
impl<T:PartitionCaptureTransport> PartitionCaptureDecoder<T> for TensorDecoder<'_, '_> {
    type Output=();
    fn decode(mut self,receipt:PartitionCaptureReceiptPlan,payload:PartitionCapturePayload<'_,T>)
        -> Result<(),PartitionCaptureExchangeError> {
        let mut received=false;
        payload.for_each(|rank,bytes| {
            if received || rank!=self.producer {return Err(PartitionCaptureExchangeError::Protocol("complete tensor producer differs"));}
            received=true;
            if self.rank==self.producer {return Ok(());}
            let index=receipt.context().selection_index;
            let prefill=self.frame.frame.prefill.is_some();
            let previous=self.frame.records().get(index).ok_or(PartitionCaptureExchangeError::Protocol("receiver record missing"))?.charged;
            let additional=if prefill {
                if previous!=self.charged { return Err(PartitionCaptureExchangeError::Protocol("remote prefill charge differs")); }
                CaptureUsage::default()
            } else {
                let difference=||Some(CaptureUsage {
                    captures:self.charged.captures.checked_sub(previous.captures)?,
                    retained_bytes:self.charged.retained_bytes.checked_sub(previous.retained_bytes)?,
                    host_bytes:self.charged.host_bytes.checked_sub(previous.host_bytes)?,
                    encoded_bytes:self.charged.encoded_bytes.checked_sub(previous.encoded_bytes)?,
                });
                difference().ok_or(PartitionCaptureExchangeError::Protocol("receiver charge exceeds complete receipt"))?
            };
            let expected = PartitionCaptureTensorReceipt {
                context:receipt.context(),identity:receipt.identity(),producer:rank,
                dtype:self.dtype.clone(),charged:self.charged,
            };
            match self.kind {
                PayloadKind::Tensor => {
                    let claim=if prefill {self.frame.take_remote_prefill_tensor(index)?} else {self.frame.take_tensor(index)?};
                    let value=claim.decode_partition_receipt(bytes,expected,self.metadata)?;
                    if prefill {self.frame.record_remote_prefill_tensor(value,self.dtype.clone())?;}
                    else {self.frame.record_tensor(value,self.dtype.clone(),additional)?;}
                }
                PayloadKind::Summary => {
                    let claim=if prefill {self.frame.take_remote_prefill_summary(index)?} else {self.frame.take_summary(index)?};
                    let value=claim.decode_partition_receipt(bytes,expected,self.metadata)?;
                    if prefill {self.frame.record_remote_prefill_summary(value,self.dtype.clone())?;}
                    else {self.frame.record_summary(value,self.dtype.clone(),additional)?;}
                }
                PayloadKind::Histogram => {
                    let claim=if prefill {self.frame.take_remote_prefill_histogram(index)?} else {self.frame.take_histogram(index)?};
                    let value=claim.decode_partition_receipt(bytes,expected,self.metadata)?;
                    if prefill {self.frame.record_remote_prefill_histogram(value,self.dtype.clone())?;}
                    else {self.frame.record_histogram(value,self.dtype.clone(),additional)?;}
                }
                PayloadKind::Candidates=>{
                    let claim=if prefill {self.frame.take_remote_prefill_candidates(index)?}else{self.frame.take_candidates(index)?};
                    let value=claim.decode_partition_receipt(bytes,expected,self.metadata)?;
                    if prefill {self.frame.record_remote_prefill_candidates(value,self.dtype.clone())?;}
                    else {self.frame.record_candidates(value,self.dtype.clone(),additional)?;}
                }
                PayloadKind::Scores=>{
                    let claim=if prefill {self.frame.take_remote_prefill_token_scores(index)?}else{self.frame.take_token_scores(index)?};
                    let value=claim.decode_partition_receipt(bytes,expected,self.metadata)?;
                    if prefill {self.frame.record_remote_prefill_token_scores(value,self.dtype.clone())?;}
                    else {self.frame.record_token_scores(value,self.dtype.clone(),additional)?;}
                }
            }
            Ok(())
        })?;
        if !received {return Err(PartitionCaptureExchangeError::Protocol("complete tensor receipt missing"));}
        if let Some(evidence)=self.evidence.take(){
            if !evidence.matches(&receipt){return Err(PartitionCaptureExchangeError::Protocol("partition evidence differs from decoded receipt"));}
            self.frame.record_partition_evidence(evidence)?;
        }
        Ok(())
    }
}
