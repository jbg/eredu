//! Borrowed source and exact destinations of ordinary coordinate projections.
use super::*;
use std::mem::{size_of, size_of_val};
use CaptureContiguousProjectionError as E;

#[derive(Debug)]
enum Source<'a> {
    Contiguous(CaptureContiguousProjectionPlan<'a>),
    Explicit { global: &'a [u64], slice: &'a ResolvedCaptureSlice, axis: usize,
        coordinates: &'a ComponentCoordinateMap },
}
/// Validated coordinate-map projection with its actual constructor population.
/// This is immutable geometry, not capture, allocation or transport authority.
#[derive(Debug)]
pub struct CaptureCoordinateProjectionPlan<'a> {
    source: Source<'a>,
    fragments: usize,
    allocation_bytes: usize,
    control_bytes: usize,
}
impl<'a> CaptureCoordinateProjectionPlan<'a> {
    /// Validate the ordinary positive-run projection without allocating storage.
    pub fn prepare(global: &'a [u64], slice: &'a ResolvedCaptureSlice, axis: usize,
        coordinates: &'a ComponentCoordinateMap, max_fragments: usize) -> Result<Self, E>
    {
        contiguous::validate(global,slice,axis,u64::try_from(coordinates.global_count()).map_err(|_|E::Overflow)?)?;
        u64::try_from(coordinates.local_count()).map_err(|_|E::Overflow)?;
        let (source,fragments,allocation_bytes,inner_controls)=if let Some(range)=coordinates.contiguous_range() {
            let plan=CaptureContiguousProjectionPlan::prepare(global,slice,axis,
                u64::try_from(range.start).map_err(|_|E::Overflow)?..u64::try_from(range.end).map_err(|_|E::Overflow)?,max_fragments)?;
            let values=(plan.fragments(),plan.allocation_bytes(),plan.control_bytes());
            (Source::Contiguous(plan),values.0,values.1,values.2)
        } else {
            let mut fragments=0usize;
            walk(slice,axis,coordinates,|_| {
                fragments=fragments.checked_add(1).ok_or(E::Overflow)?;
                if fragments>max_fragments {return Err(E::Fragments);} Ok(())
            })?;
            let vectors=6usize.checked_add(8usize.checked_mul(fragments).ok_or(E::Overflow)?).ok_or(E::Overflow)?;
            let allocation=global.len().checked_mul(size_of::<u64>()).and_then(|n|n.checked_mul(vectors))
                .and_then(|n|n.checked_add(size_of::<CaptureFragmentGeometry>().checked_mul(fragments)?)).ok_or(E::Overflow)?;
            (Source::Explicit{global,slice,axis,coordinates},fragments,allocation,0)
        };
        let parts=[size_of::<Self>()*2,size_of::<Source<'_>>(),size_of::<CaptureSlicePartition>()*2,
            size_of::<Run>()*2,size_of::<Option<Run>>(),size_of::<[u64;8]>(),size_of::<[usize;5]>(),
            size_of::<ResolvedCaptureSlice>()*2,size_of::<CaptureFragmentGeometry>(),
            size_of::<Vec<CaptureFragmentGeometry>>(),size_of::<Vec<u64>>(),
            size_of::<(&[u64],&ResolvedCaptureSlice,usize,&ComponentCoordinateMap,usize)>(),
            size_of::<(&ResolvedCaptureSlice,usize,&ComponentCoordinateMap)>(),
            size_of::<Result<Self,E>>(),size_of::<Result<(),E>>(),size_of::<std::ops::Range<usize>>(),
            size_of::<(&mut CaptureSlicePartition,usize)>(),size_of::<(&mut usize,usize)>(),
            size_of::<[&Vec<u64>;4]>(),inner_controls];
        let control_bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or(E::Overflow)?;
        allocation_bytes.checked_add(control_bytes).ok_or(E::Overflow)?;
        Ok(Self{source,fragments,allocation_bytes,control_bytes})
    }
    /// Exact requested owning Vec storage, including each fragment's two slices.
    pub const fn allocation_bytes(&self)->usize {self.allocation_bytes}
    /// Named fixed validation, traversal, construction and return controls.
    pub const fn control_bytes(&self)->usize {self.control_bytes}
    /// Checked sum to reserve before constructing the owning projection.
    pub const fn requested_bytes(&self)->usize {self.allocation_bytes+self.control_bytes}
    /// Exact number of positive arithmetic-run fragments.
    pub const fn fragments(&self)->usize {self.fragments}
    /// Construct through the same run worker used by ordinary projections.
    pub fn construct(self)->CaptureSlicePartition {
        match self.source {
            Source::Contiguous(source)=>source.construct(),
            Source::Explicit{global,slice,axis,coordinates}=> {
                let mut local_shape=global.to_vec();
                local_shape[axis]=coordinates.local_count() as u64;
                let mut plan=CaptureSlicePartition{global_shape:global.to_vec(),local_shape,
                    global_slice:slice.clone(),axis,fragments:Vec::with_capacity(self.fragments)};
                walk(slice,axis,coordinates,|run| {
                    plan.push(run.0,run.1,run.2,run.3,run.4,run.5,self.fragments)
                        .expect("retained checked coordinate projection"); Ok(())
                }).expect("retained checked coordinate traversal");
                plan
            }
        }
    }
}
type Run=(u64,u64,u64,u64,u64,u64,usize);
// Coalesce positive arithmetic runs in local and destination order. Descending
// and permuted destinations remain separate; no unrelated column is exported.
fn walk(slice:&ResolvedCaptureSlice,axis:usize,coordinates:&ComponentCoordinateMap,
    mut emit:impl FnMut(Run)->Result<(),E>)->Result<(),E>
{
    if elements(&slice.shape).map_err(|_|E::Overflow)?==0 {return Ok(());}
    let (start,end,stride)=(slice.starts[axis],slice.ends[axis],slice.strides[axis]);
    let mut run:Option<Run>=None;
    for local in 0..coordinates.local_count() {
        let global=u64::try_from(coordinates.local_to_global(local).expect("validated coordinate map")).map_err(|_|E::Overflow)?;
        if global<start||global>=end||!(global-start).is_multiple_of(stride) {continue;}
        let destination=(global-start)/stride;
        let local=u64::try_from(local).map_err(|_|E::Overflow)?;
        if let Some((first_local,last_local,local_step,first_destination,last_destination,destination_step,count))=run {
            if destination>last_destination && (count==1 || (local-last_local==local_step && destination-last_destination==destination_step)) {
                run=Some((first_local,local,local-last_local,first_destination,destination,destination-last_destination,count.checked_add(1).ok_or(E::Overflow)?));
                continue;
            }
            emit(run.expect("present run"))?;
        }
        run=Some((local,local,1,destination,destination,1,1));
    }
    if let Some(run)=run {emit(run)?;}
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_coordinate_projection_prices_permuted_runs_and_preserves_stride_origins() {
        let global=[2,2,10];
        let slice=ResolvedCaptureSlice{starts:vec![0,0,1],ends:vec![2,2,10],strides:vec![1,1,2],shape:vec![2,2,5]};
        for values in [vec![9,1,3,7],vec![]] {
            let coordinates=ComponentCoordinateMap::indices(10,values).unwrap();
            let source=CaptureCoordinateProjectionPlan::prepare(&global,&slice,2,&coordinates,3).unwrap();
            let count=source.fragments();
            assert_eq!(count,if coordinates.local_count()==0 {0} else {3});
            let bytes=source.allocation_bytes();
            let result=source.construct();
            let slices=std::iter::once(&result.global_slice).chain(result.fragments.iter().flat_map(|f|[&f.local,&f.destination]));
            let words=slices.map(|s|s.starts.capacity()+s.ends.capacity()+s.strides.capacity()+s.shape.capacity()).sum::<usize>()
                +result.global_shape.capacity()+result.local_shape.capacity();
            assert_eq!(bytes,words*size_of::<u64>()+result.fragments.capacity()*size_of::<CaptureFragmentGeometry>());
            if count!=0 {
                assert_eq!(result.fragments[0].destination.starts,[0,0,4]);
                assert_eq!(result.fragments[1].local.starts,[0,0,1]);
                assert_eq!(result.fragments[1].destination.shape,[2,2,2]);
                assert_eq!(result.fragments[2].destination.starts,[0,0,3]);
                assert_eq!(CaptureCoordinateProjectionPlan::prepare(&global,&slice,2,&coordinates,2).unwrap_err(),E::Fragments);
            }
        }
    }
}
