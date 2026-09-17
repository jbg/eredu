//! Exact borrowed completed-record encoding into an original counted destination.
use super::*;
use eredu_core::capture::{BorrowedPartitionCaptureFragmentRecord, BorrowedPartitionCaptureProducerRecord};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Borrowed encoding input. It is descriptive only: placement, record validity,
/// native completion and transaction acceptance remain the enclosing driver's
/// responsibility. The fixed completed-record source has no callback escape.
#[derive(Debug)]
pub struct PartitionCaptureRecordEncoding<'a> {
    context: &'a PartitionCaptureContext,
    identity: &'a str,
    rank: usize,
    combination: PartitionCaptureCombination,
    record: &'a CaptureRecord,
    maximum: usize,
}
/// Encoding rejection retains both original error storage and its account.
#[derive(Debug, thiserror::Error)]
pub enum PartitionCaptureEncodingError {
    /// Fixed source/schema check failed before output allocation.
    #[error("partition capture encoding source: {reason}")]
    Source {
        /// Actual closed-source refusal.
        reason: &'static str,
        /// Original account outliving the failure.
        funding: WorkspaceMetadataFunding,
    },
    /// Fixed writer controls could not be reserved.
    #[error("partition capture writer reservation: {cause}")]
    Funding {
        /// Original metadata failure.
        #[source]
        cause: WorkspaceMetadataFundingError,
        /// Nonrefunding original account.
        funding: WorkspaceMetadataFunding,
    },
    /// Counted output destination failed.
    #[error(transparent)]
    Destination(#[from] PartitionCaptureStorageError),

}
impl<'a> PartitionCaptureRecordEncoding<'a> {
    /// Lend an already produced canonical record. No clone, allocation, native
    /// operation, metadata reservation or producer certification occurs here.
    pub fn new(
        context: &'a PartitionCaptureContext,
        identity: &'a str,
        rank: usize,
        combination: PartitionCaptureCombination,
        record: &'a CaptureRecord,
        maximum: usize,
    ) -> Self { Self { context, identity, rank, combination, record, maximum } }

    /// Encode one complete global-producer record with the canonical core wire
    /// serializer. Original H owns the record; the supplied metadata source pays
    /// output/serializer storage separately. No raw protected tensor exits.
    pub fn encode(self, funding: &WorkspaceMetadataFunding)
        -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureEncodingError>
    {
        let source = |reason| PartitionCaptureEncodingError::Source { reason, funding: funding.clone() };
        let controls = [
            size_of::<Self>(), eredu_core::capture::CaptureRecordWire::control_bytes(),
            size_of::<BorrowedPartitionCaptureProducerRecord<'_, [BorrowedPartitionCaptureFragmentRecord<'_>; 1]>>(),
            size_of::<BorrowedPartitionCaptureFragmentRecord<'_>>(),
            size_of::<(&Self, &WorkspaceMetadataFunding)>(),
            size_of::<Result<PartitionCaptureBuffer<u8>, PartitionCaptureEncodingError>>(),
        ];
        let controls = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(||source("writer controls overflow"))?;
        funding.reserve_metadata(controls).map_err(|cause|PartitionCaptureEncodingError::Funding { cause, funding: funding.clone() })?;
        if self.maximum == 0 || self.identity.is_empty()
            || self.context.selection_index == usize::MAX
            || self.combination != PartitionCaptureCombination::Disjoint
        { return Err(source("complete producer receipt source differs")); }
        let fragments = [BorrowedPartitionCaptureFragmentRecord { fragment_index: 0, record: self.record }];
        let record = BorrowedPartitionCaptureProducerRecord {
            schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
            combination: self.combination,
            receipt_plan_identity: self.identity,
            context: self.context,
            producer_rank: self.rank,
            source_dtype: self.record.source_dtype.as_ref(),
            fragments,
        };
        encode_closed(&record, self.maximum, funding)
    }
}
/// Runtime-only counterpart for an authenticated contiguous source. Its exact
/// zero/one fragment and payload come from the paid fragment bank. It grants no
/// authority to arbitrary Serialize callbacks or caller-created wire values.
pub(crate) fn encode_contiguous(
    context: &PartitionCaptureContext, identity: &str, rank: usize,
    combination: PartitionCaptureCombination, dtype: Option<&TensorDtype>,
    record: Option<&eredu_core::capture::CaptureRecordWire<'_>>, maximum: usize,
    funding: &WorkspaceMetadataFunding,
) -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureEncodingError> {
    let source = |reason| PartitionCaptureEncodingError::Source { reason, funding: funding.clone() };
    type Fragment<'a> = BorrowedPartitionCaptureFragmentRecord<'a, eredu_core::capture::CaptureRecordWire<'a>>;
    let parts = [
        size_of::<Option<&eredu_core::capture::CaptureRecordWire<'_>>>(),
        size_of::<[Option<Fragment<'_>>; 1]>(),
        size_of::<BorrowedPartitionCaptureProducerRecord<'_, Fragments<'_>>>(),
        size_of::<Fragments<'_>>(), eredu_core::capture::CaptureRecordWire::control_bytes(),
        size_of::<(&PartitionCaptureContext, &str, usize, PartitionCaptureCombination,
            Option<&TensorDtype>, usize, &WorkspaceMetadataFunding)>(),
    ];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or_else(||source("fragment writer controls overflow"))?)
        .map_err(|cause|PartitionCaptureEncodingError::Funding{cause,funding:funding.clone()})?;
    if maximum == 0 || identity.is_empty() || context.selection_index == usize::MAX
        || record.is_some_and(|r| r.source_dtype != dtype) {
        return Err(source("contiguous producer receipt source differs"));
    }
    let fragments = [record.map(|record| BorrowedPartitionCaptureFragmentRecord { fragment_index: 0, record })];
    let envelope = BorrowedPartitionCaptureProducerRecord {
        schema_version: PARTITION_CAPTURE_SCHEMA_VERSION, combination,
        receipt_plan_identity: identity, context, producer_rank: rank, source_dtype: dtype,
        fragments: Fragments(&fragments),
    };
    encode_closed(&envelope, maximum, funding)
}
// The canonical sequence serializer still owns framing; this lends zero/one
// fixed records without allocating a Vec or emitting a null fragment.
struct Fragments<'a>(&'a [Option<BorrowedPartitionCaptureFragmentRecord<'a, eredu_core::capture::CaptureRecordWire<'a>>>; 1]);
impl serde::Serialize for Fragments<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(usize::from(self.0[0].is_some())))?;
        if let Some(fragment) = &self.0[0] { sequence.serialize_element(fragment)?; }
        sequence.end()
    }
}
/// Closed multi-fragment counterpart. Each immutable borrowed record comes from
/// the authenticated original fragment bank; no arbitrary callback is accepted.
pub(crate) fn encode_fragment_records(context:&PartitionCaptureContext,identity:&str,rank:usize,
    combination:PartitionCaptureCombination,dtype:Option<&TensorDtype>,
    records:&[eredu_core::capture::CaptureRecordWire<'_>],maximum:usize,funding:&WorkspaceMetadataFunding)
    ->Result<PartitionCaptureBuffer<u8>,PartitionCaptureEncodingError> {
    let source=|reason|PartitionCaptureEncodingError::Source{reason,funding:funding.clone()};
    let parts=[size_of::<RecordFragments<'_,'_>>(),
        size_of::<BorrowedPartitionCaptureProducerRecord<'_,RecordFragments<'_,'_>>>(),
        size_of::<BorrowedPartitionCaptureFragmentRecord<'_,eredu_core::capture::CaptureRecordWire<'_>>>(),
        eredu_core::capture::CaptureRecordWire::control_bytes(),
        size_of::<(&PartitionCaptureContext,&str,usize,PartitionCaptureCombination,Option<&TensorDtype>,&[eredu_core::capture::CaptureRecordWire<'_>],usize,&WorkspaceMetadataFunding)>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_,eredu_core::capture::CaptureRecordWire<'_>>>>(),
        size_of::<(usize,&eredu_core::capture::CaptureRecordWire<'_>)>(),
        size_of::<Result<PartitionCaptureBuffer<u8>,PartitionCaptureEncodingError>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or_else(||source("fragment writer controls overflow"))?)
        .map_err(|cause|PartitionCaptureEncodingError::Funding{cause,funding:funding.clone()})?;
    if maximum==0||identity.is_empty()||context.selection_index==usize::MAX
        ||records.iter().any(|record|record.source_dtype!=dtype) {
        return Err(source("multi-fragment receipt source differs"));
    }
    let envelope=BorrowedPartitionCaptureProducerRecord{schema_version:PARTITION_CAPTURE_SCHEMA_VERSION,
        combination,receipt_plan_identity:identity,context,producer_rank:rank,source_dtype:dtype,
        fragments:RecordFragments(records)};
    encode_closed(&envelope,maximum,funding)
}
struct RecordFragments<'r,'a>(&'r [eredu_core::capture::CaptureRecordWire<'a>]);
impl serde::Serialize for RecordFragments<'_,'_> {
    fn serialize<S:serde::Serializer>(&self,serializer:S)->Result<S::Ok,S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence=serializer.serialize_seq(Some(self.0.len()))?;
        for (fragment_index,record) in self.0.iter().enumerate() {
            sequence.serialize_element(&BorrowedPartitionCaptureFragmentRecord{fragment_index,record})?;
        }
        sequence.end()
    }
}

// Private shared writer. Only the closed record forms above can reach it.
fn encode_closed<T: serde::Serialize>(record: &T, maximum: usize, funding: &WorkspaceMetadataFunding)
    -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureEncodingError> {
    let source = |reason| PartitionCaptureEncodingError::Source { reason, funding: funding.clone() };
    let parts = [size_of::<(&T,usize,&WorkspaceMetadataFunding)>(),size_of::<Destination<'_>>(),
        crate::capture::RECORD_ENCODING_CONTROL_BYTES, size_of::<serde_json::Serializer<&mut Destination<'_>>>(),
        size_of::<Option<u64>>(), size_of::<usize>(), size_of::<bool>(),
        size_of::<PartitionCaptureEncodingError>(),size_of::<Result<PartitionCaptureBuffer<u8>,PartitionCaptureEncodingError>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or_else(||source("closed writer controls overflow"))?)
        .map_err(|cause|PartitionCaptureEncodingError::Funding{cause,funding:funding.clone()})?;
    let count = crate::capture::encoded::encoded_size(record)
        .and_then(|n|usize::try_from(n).ok()).filter(|n|*n <= maximum)
        .ok_or_else(||source("receipt exceeds its encoded bound"))?;
    let mut output = PartitionCaptureBuffer::funded(count, funding)?;
    let mut writer = Destination { output: &mut output, fits: true };
    let written = serde_json::to_writer(&mut writer, record).is_ok();
    if !written || !writer.fits || writer.output.len() != count { return Err(source("writer count changed")); }
    Ok(output)
}

struct Destination<'a> { output: &'a mut PartitionCaptureBuffer<u8>, fits: bool }
impl std::io::Write for Destination<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        // A read-only source was just measured. Still check every write rather
        // than grow if that source contract changes. Like Counter, accepting the
        // write avoids allocating an error merely to report a fixed overflow.
        self.fits &= self.output.extend_from_slice(bytes).is_ok();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

#[cfg(test)]
mod tests;
