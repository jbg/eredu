//! Selected writer copies borrow the already settled five-source scope.
//! Keep alternative transfer temporaries off the live provider callback stack.
use super::*;
use eredu_runtime::working_memory::WorkingMemoryStorage;

#[inline(never)]
pub(super) fn dispatch<'s>(source:&RoutedUnitCaptureSource<'s,Array>,completed:[safemlx::EvaluatedArray<'s>;5],
    bank:eredu_core::capture::RoutedUnitGeometry,tokens:u64,writer:RoutedWriter<'_,'_,'_,'_>,
    scope:&mut WorkingMemoryFundingScope,segment:Option<&mut CaptureSourceSegment>,
    source_pin:WorkingMemoryStorage<StorageIdentity>)->Result<(),Error>{
    match writer {
        RoutedWriter::Prefill(writer)=>prefill(source,completed,bank,tokens,writer,scope,
            segment.ok_or_else(||error(CaptureCarrierError::Identity))?,source_pin),
        RoutedWriter::Invocation(writer)=>invocation(source,completed,bank,tokens,writer,scope,source_pin),
        RoutedWriter::Partition{writer,origins,units,..}=>partition(source,completed,writer,origins,units,scope,source_pin),
    }
}
#[inline(never)]
fn prefill<'s>(source:&RoutedUnitCaptureSource<'s,Array>,completed:[safemlx::EvaluatedArray<'s>;5],
    bank:eredu_core::capture::RoutedUnitGeometry,tokens:u64,writer:CaptureRoutedPrefillWriter<'_,'_,'_,'_>,
    scope:&mut WorkingMemoryFundingScope,segment:&mut CaptureSourceSegment,
    source_pin:WorkingMemoryStorage<StorageIdentity>)->Result<(),Error>{
    let prepared=CompletedRoutedCaptureSource::new(source,completed,bank,tokens).map_err(error)?;
    let transfer=RoutedCaptureTransfer::Prefill(writer.prepare_with_segment_source(scope,segment,source_pin).map_err(error)?);
    prepared.copy_routed(transfer).map_err(error)
}
#[inline(never)]
fn invocation<'s>(source:&RoutedUnitCaptureSource<'s,Array>,completed:[safemlx::EvaluatedArray<'s>;5],
    bank:eredu_core::capture::RoutedUnitGeometry,tokens:u64,writer:CaptureRoutedBatchWriter<'_,'_>,
    scope:&mut WorkingMemoryFundingScope,source_pin:WorkingMemoryStorage<StorageIdentity>)->Result<(),Error>{
    let prepared=CompletedRoutedCaptureSource::new(source,completed,bank,tokens).map_err(error)?;
    let transfer=RoutedCaptureTransfer::Invocation(writer.prepare_with_source(scope,source_pin).map_err(error)?);
    prepared.copy_routed(transfer).map_err(error)
}
#[inline(never)]
fn partition<'s>(source:&RoutedUnitCaptureSource<'s,Array>,completed:[safemlx::EvaluatedArray<'s>;5],
    writer:eredu_runtime::working_memory::CapturePartitionRoutedWriter<'_,'_>,
    origins:Option<eredu_core::capture::RoutedUnitOrigins<'_>>,units:&eredu_core::component::ComponentCoordinateMap,
    scope:&mut WorkingMemoryFundingScope,source_pin:WorkingMemoryStorage<StorageIdentity>)->Result<(),Error>{
    let layout=writer.source_layout();
    let prepared=CompletedPartitionRoutedCaptureSource::bind_layout(source,completed,
        PartitionRoutedCaptureLayout{geometry:layout.geometry,source_tokens:layout.source_tokens,
            ownership:layout.ownership,origins,units},writer.request().source_tokens).map_err(error)?;
    let transfer=writer.prepare_with_source(scope,source_pin).map_err(error)?;
    prepared.copy_routed(transfer).map_err(error)
}
