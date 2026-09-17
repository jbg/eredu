//! Exact component maps retained by the existing dense fragment program.
use super::*;
use eredu_core::component::ComponentCoordinateMap;
#[derive(Clone,Copy)]
pub(super) enum Sources<'a> {
    Contiguous(&'a [PartitionCaptureContiguousProducer]),
    Coordinates(&'a [PartitionCaptureCoordinateProducer<'a>]),
}
#[derive(Debug)]
pub(super) enum Owned {
    Contiguous(Vec<PartitionCaptureContiguousProducer>),
    Coordinates(Vec<(usize,ComponentCoordinateMap)>),
}
impl Sources<'_> {
    pub(super) fn len(self)->usize {match self {Self::Contiguous(rows)=>rows.len(),Self::Coordinates(rows)=>rows.len()}}
    pub(super) fn is_empty(self)->bool {self.len()==0}
    pub(super) fn contiguous(self)->bool {matches!(self,Self::Contiguous(_))}
    pub(super) fn rank(self,index:usize)->usize {match self {Self::Contiguous(rows)=>rows[index].rank,Self::Coordinates(rows)=>rows[index].rank}}
    pub(super) fn copy(self,metadata:&WorkspaceMetadataFunding)->Result<Owned,Cause> {match self {
        Self::Contiguous(rows)=>{
            let mut owned=metadata.metadata_vec(rows.len())?;
            for row in rows {
                if row.coordinates.start>row.coordinates.end {return Err(Cause::Source("contiguous rank source differs"));}
                owned.push(row.clone());
            }
            Ok(Owned::Contiguous(owned))
        },
        Self::Coordinates(rows)=>{
            let mut owned=metadata.metadata_vec(rows.len())?;
            for row in rows {
                metadata.reserve_metadata(ComponentCoordinateMap::copy_control_bytes().ok_or(Cause::Source("component copy controls overflow"))?)?;
                let count=row.coordinates.copy_storage_elements();
                let map=row.coordinates.copy_with_storage(metadata.metadata_vec(count)?,metadata.metadata_vec(count)?)
                    .map_err(|_|Cause::Source("paid component coordinate destination differs"))?;
                owned.push((row.rank,map));
            }
            Ok(Owned::Coordinates(owned))
        },
    }}
}
impl Owned {
    pub(super) fn minimum_rank(&self)->Option<usize> {match self {
        Self::Contiguous(rows)=>rows.iter().map(|row|row.rank).min(),
        Self::Coordinates(rows)=>rows.iter().map(|row|row.0).min(),
    }}
    pub(super) fn receipt(&self,source:&SharedCapturePlan,context:&PartitionCaptureContext,axis:usize,
        combination:PartitionCaptureCombination,world:usize,limits:PartitionCaptureReceiptLimits,
        metadata:&WorkspaceMetadataFunding,ledger:&mut dyn CaptureReservation)
        ->Result<PartitionCaptureReceiptPlan,PartitionCaptureReceiptConstructionError> {
        match self {
            Self::Contiguous(rows)=>PartitionCaptureReceiptPlan::new_contiguous_shared_funded(source,context,
                axis,rows,combination,world,limits,metadata,ledger),
            Self::Coordinates(rows)=>{
                // The borrowed adapter is a real paid destination. Every map
                // remains owned here through construction and any refusal.
                let mut borrowed=metadata.metadata_vec(rows.len()).map_err(|cause|
                    super::super::super::receipt::component_adapter_error(source,metadata,cause))?;
                for (rank,coordinates) in rows {borrowed.push(PartitionCaptureCoordinateProducer{rank:*rank,coordinates});}
                PartitionCaptureReceiptPlan::new_coordinates_shared_funded(source,context,axis,&borrowed,
                    combination,world,limits,metadata,ledger)
            },
        }
    }
}
impl PreparedPartitionContiguousSource {
    /// Exact component maps with canonical prefill windows, using the existing
    /// source vote, fragment allowance, Host writer and delivery program.
    pub fn new_local_coordinates(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureCoordinateProducer<'_>],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        inference:InferenceGeometry,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        Self::coordinates(source,index,axis,producers,native,local,combination,Coordinate::Prefill(inference),metadata)
    }
    /// One actual complete invocation, including a terminal prefill hook. This
    /// carries no prefill chunk or synthetic temporal source.
    pub fn new_local_coordinates_invocation(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureCoordinateProducer<'_>],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        phase:CapturePhase,prediction:u64,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        Self::coordinates(source,index,axis,producers,native,local,combination,Coordinate::Invocation(phase,prediction),metadata)
    }
    fn coordinates(source:&SharedCapturePlan,index:usize,axis:usize,
        producers:&[PartitionCaptureCoordinateProducer<'_>],native:&[PartitionCaptureFragmentGeometry<'_>],
        local:Option<PartitionCaptureLocalSource<'_>>,combination:PartitionCaptureCombination,
        coordinate:Coordinate,metadata:&WorkspaceMetadataFunding)->Result<Self,PartitionCaptureProgramError> {
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        metadata.reserve_metadata(control_bytes().ok_or_else(||error(Cause::Source("component constructor controls overflow")))?)
            .map_err(|cause|error(cause.into()))?;
        let mut dtypes=metadata.metadata_vec(producers.len()).map_err(|cause|error(cause.into()))?;
        for producer in producers {dtypes.push(local.as_ref().filter(|local|local.producer==producer.rank).map(|local|local.dtype.clone()));}
        Self::new_sources(source,index,axis,Sources::Coordinates(producers),&dtypes,voted::Sources::Geometry(native),
            combination,coordinate,local,metadata)
    }
}
pub(super) fn control_bytes()->Option<usize> {
    let frames=[size_of::<Sources<'_>>()*2,size_of::<Owned>()*2,size_of::<Result<Owned,Cause>>(),
        size_of::<ComponentCoordinateMap>()*2,size_of::<(usize,ComponentCoordinateMap)>(),
        size_of::<PartitionCaptureCoordinateProducer<'_>>()*2,size_of::<Vec<PartitionCaptureCoordinateProducer<'_>>>(),
        size_of::<Result<Vec<PartitionCaptureCoordinateProducer<'_>>,eredu_nn::Error>>(),
        size_of::<(&SharedCapturePlan,usize,usize,&[PartitionCaptureCoordinateProducer<'_>],&[PartitionCaptureFragmentGeometry<'_>],
            Option<PartitionCaptureLocalSource<'_>>,PartitionCaptureCombination,Coordinate,&WorkspaceMetadataFunding)>(),
        size_of::<(&Owned,&SharedCapturePlan,&PartitionCaptureContext,usize,PartitionCaptureCombination,usize,
            PartitionCaptureReceiptLimits,&WorkspaceMetadataFunding,&mut dyn CaptureReservation)>(),
        size_of::<std::slice::Iter<'_,(usize,ComponentCoordinateMap)>>(),
        size_of::<std::slice::Iter<'_,PartitionCaptureCoordinateProducer<'_>>>(),
        ComponentCoordinateMap::copy_control_bytes()?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
