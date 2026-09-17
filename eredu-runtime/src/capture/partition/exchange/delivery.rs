//! Shared canonical payload reconstruction; original consumers keep paid owners.
use super::*;
use std::mem::{size_of, size_of_val};

pub(crate) trait PartitionCaptureDecoder<T: PartitionCaptureTransport> {
    type Output;
    fn decode(self, receipt:PartitionCaptureReceiptPlan, payload:PartitionCapturePayload<'_,T>)
        -> Result<Self::Output,PartitionCaptureExchangeError>;
}

pub(crate) struct PartitionCapturePayload<'a,T:PartitionCaptureTransport> {
    transport:&'a T,
    headers:&'a [u32],
    received:&'a [u32],
    width:usize,
    local_rank:usize,
    local:Option<&'a [u8]>,
}
impl<'a,T:PartitionCaptureTransport> PartitionCapturePayload<'a,T> {
    pub(super) fn new(transport:&'a T,headers:&'a [u32],received:&'a [u32],width:usize,
        local_rank:usize,local:Option<&'a [u8]>) -> Self {
        Self {transport,headers,received,width,local_rank,local}
    }
    /// Padding and the local source are checked before each borrowed receipt is
    /// lent. Scratch stays owned by its counted destination until the loan ends.
    pub(crate) fn for_each(self, mut receive:impl FnMut(usize,&[u8])->Result<(),PartitionCaptureExchangeError>)
        -> Result<(),PartitionCaptureExchangeError> {
        if self.width==0 {return Err(PartitionCaptureExchangeError::Protocol("empty receipt payload"));}
        for (rank,frame) in self.received.chunks_exact(self.width).enumerate() {
            let header=&self.headers[rank*HEADER_WORDS..(rank+1)*HEADER_WORDS];
            let length=usize::try_from(header[14] as u64 | ((header[15] as u64)<<32))
                .map_err(|_|CaptureError::Overflow)?;
            let mut bytes=self.transport.capture_byte_destination(self.width.checked_mul(4).ok_or(CaptureError::Overflow)?)?;
            for word in frame {bytes.extend_from_slice(&word.to_le_bytes())?;}
            if length>bytes.len() || bytes[length..].iter().any(|byte|*byte!=0) {
                return Err(PartitionCaptureExchangeError::Protocol("payload padding"));
            }
            if rank==self.local_rank && self.local.is_some_and(|local|local!=&bytes[..length]) {
                return Err(PartitionCaptureExchangeError::Protocol("local receipt changed in transport"));
            }
            if header[13]==2 {receive(rank,&bytes[..length])?;}
        }
        Ok(())
    }
}

pub(super) struct OrdinaryDecoder<'a> {pub ledger:&'a mut dyn CaptureReservation}
impl<T:PartitionCaptureTransport> PartitionCaptureDecoder<T> for OrdinaryDecoder<'_> {
    type Output=ReceivedPartitionCapture;
    fn decode(self,receipt:PartitionCaptureReceiptPlan,payload:PartitionCapturePayload<'_,T>)
        -> Result<Self::Output,PartitionCaptureExchangeError> {
        let mut delivery=receipt.into_delivery();
        payload.for_each(|rank,bytes| {delivery.receive(rank,bytes,self.ledger)?;Ok(())})?;
        Ok(delivery.finish(self.ledger)?)
    }
}

impl<T:PartitionCaptureTransport> PartitionCaptureExchange<'_,T> {
    /// Named storage for the shared protocol and one selected closed decoder.
    /// Payload allocations and decoder-specific parser/host storage are separate.
    pub(crate) fn decoder_control_bytes<D:PartitionCaptureDecoder<T>>() -> Option<usize> {
        let frames=[size_of::<Self>(),size_of::<D>(),size_of::<D::Output>(),
            size_of::<PartitionCaptureReceiptPlan>(),size_of::<PartitionCapturePayload<'_,T>>(),
            size_of::<[u32;HEADER_WORDS]>()*2,size_of::<[u8;4]>(),
            size_of::<PartitionCaptureBuffer<u32>>()*4,size_of::<PartitionCaptureBuffer<u8>>(),
            size_of::<Option<PartitionCaptureBuffer<u8>>>(),
            size_of::<Result<Option<PartitionCaptureBuffer<u8>>,PartitionCaptureExchangeError>>(),
            size_of::<Result<D::Output,PartitionCaptureExchangeError>>()*2,
            size_of::<PartitionCaptureExchangeError>()*2,
            size_of::<Box<PartitionCaptureExchangeError>>(),
            size_of::<Option<PartitionCaptureExchangeError>>(),
            size_of::<(&T,BoundedCompletionWait,usize,usize,u64,Option<usize>)>(),
            size_of::<std::slice::ChunksExact<'_,u32>>()*2,
            size_of::<std::slice::Chunks<'_,u8>>(),
            // Exact fixed diagnostic allocation in the existing local source check.
            "local capture receipt presence or byte bound".len()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
