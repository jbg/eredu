//! Retained, allocation-free source for the ordinary contiguous constructor.
use super::*;
use std::mem::{size_of, size_of_val};

/// Validated immutable inputs and exact storage for one contiguous projection.
/// This describes host geometry only; it grants no capture, native, transport,
/// or allocation permission. An original caller pays these populations first.
#[derive(Debug)]
pub struct CaptureContiguousProjectionPlan<'a> {
    global: &'a [u64],
    slice: &'a ResolvedCaptureSlice,
    axis: usize,
    range: Range<u64>,
    fragments: usize,
    allocation_bytes: usize,
    control_bytes: usize,
}
impl<'a> CaptureContiguousProjectionPlan<'a> {
    /// Check the same ordinary projection equations without allocating a map,
    /// shape, slice, fragment vector, or diagnostic string.
    pub fn prepare(global:&'a [u64],slice:&'a ResolvedCaptureSlice,axis:usize,
        range:Range<u64>,max_fragments:usize) -> Result<Self,CaptureContiguousProjectionError> {
        use CaptureContiguousProjectionError as E;
        let count=global.get(axis).copied().ok_or(E::Axes)?;
        validate(global,slice,axis,count)?;
        if range.start>range.end || range.end>count {return Err(E::Range);}
        let fragments=usize::from(elements(&slice.shape).map_err(|_|E::Overflow)?!=0
            && span(slice,axis,range.clone()).is_some());
        if fragments>max_fragments {return Err(E::Fragments);}
        let rank=global.len();
        // Global/local shape + original slice; then both fragment slices.
        let vectors=6usize.checked_add(8usize.checked_mul(fragments).ok_or(E::Overflow)?).ok_or(E::Overflow)?;
        let allocation_bytes=rank.checked_mul(size_of::<u64>()).and_then(|n|n.checked_mul(vectors))
            .and_then(|n|n.checked_add(size_of::<CaptureFragmentGeometry>().checked_mul(fragments)?))
            .ok_or(E::Overflow)?;
        let parts=[size_of::<Self>()*2,size_of::<CaptureSlicePartition>()*2,
            size_of::<CaptureFragmentGeometry>(),size_of::<ResolvedCaptureSlice>()*2,
            size_of::<Vec<CaptureFragmentGeometry>>(),size_of::<Vec<u64>>(),
            size_of::<Result<Self,E>>(),size_of::<Range<u64>>(),size_of::<Option<Span>>(),
            size_of::<[&Vec<u64>;4]>(),size_of::<(&[u64],&ResolvedCaptureSlice,usize,usize)>(),
            CaptureSlicePartition::contiguous_projection_control_bytes().ok_or(E::Overflow)?];
        let control_bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or(E::Overflow)?;
        allocation_bytes.checked_add(control_bytes).ok_or(E::Overflow)?;
        Ok(Self {global,slice,axis,range,fragments,allocation_bytes,control_bytes})
    }
    /// Exact requested Vec backing bytes for this immutable constructor source.
    pub const fn allocation_bytes(&self)->usize {self.allocation_bytes}
    /// Named source, validation, construction, and return controls.
    pub const fn control_bytes(&self)->usize {self.control_bytes}
    /// Checked sum to reserve before calling the ordinary owning constructor.
    pub const fn requested_bytes(&self)->usize {self.allocation_bytes+self.control_bytes}
    /// Exact native-fragment geometry count, zero when the selection has no overlap.
    pub const fn fragments(&self)->usize {self.fragments}
    /// Build only the validated destinations, without geometric reinterpretation
    /// or growth. The same private worker fills ordinary and original projections.
    pub fn construct(self)->CaptureSlicePartition {
        let rank=self.global.len();
        let mut local_shape=self.global.to_vec();
        local_shape[self.axis]=self.range.end-self.range.start;
        let mut fragments=Vec::with_capacity(self.fragments);
        if self.fragments!=0 {
            let mut local=ResolvedCaptureSlice {starts:vec![0;rank],ends:vec![0;rank],
                strides:vec![0;rank],shape:vec![0;rank]};
            let mut destination=ResolvedCaptureSlice {starts:vec![0;rank],ends:vec![0;rank],
                strides:vec![0;rank],shape:vec![0;rank]};
            let projected=CaptureSlicePartition::contiguous_fragment_into(self.global,self.slice,
                self.axis,self.range,&mut local,&mut destination).expect("retained checked projection");
            assert!(projected,"retained nonempty projection");
            fragments.push(CaptureFragmentGeometry {local,destination});
        }
        CaptureSlicePartition {global_shape:self.global.to_vec(),local_shape,
            global_slice:self.slice.clone(),axis:self.axis,fragments}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_constructor_source_prices_actual_storage_and_preserves_global_stride() {
        let global=[2,9];
        let slice=ResolvedCaptureSlice {starts:vec![0,1],ends:vec![2,9],strides:vec![1,3],shape:vec![2,3]};
        for (range,expected_fragments) in [(3..8,1),(0..1,0)] {
            let source=CaptureContiguousProjectionPlan::prepare(&global,&slice,1,range,1).unwrap();
            assert_eq!(source.fragments(),expected_fragments);
            let bytes=source.allocation_bytes();
            assert!(source.requested_bytes()>bytes);
            let projection=source.construct();
            let slices=std::iter::once(&projection.global_slice).chain(projection.fragments.iter().flat_map(|f|[&f.local,&f.destination]));
            let capacities=slices.map(|s|s.starts.capacity()+s.ends.capacity()+s.strides.capacity()+s.shape.capacity()).sum::<usize>()
                +projection.global_shape.capacity()+projection.local_shape.capacity();
            assert_eq!(bytes,capacities*size_of::<u64>()+projection.fragments.capacity()*size_of::<CaptureFragmentGeometry>());
            if expected_fragments==1 {
                let fragment=&projection.fragments()[0];
                assert_eq!(projection.local_shape(),&[2,5]);
                assert_eq!(fragment.local().starts,&[0,1]);
                assert_eq!(fragment.local().ends,&[2,5]);
                assert_eq!(fragment.local().strides,&[1,3]);
                assert_eq!(fragment.destination().starts,&[0,1]);
                assert_eq!(fragment.destination().shape,&[2,2]);
            }
        }
        assert_eq!(CaptureContiguousProjectionPlan::prepare(&global,&slice,1,3..8,0).unwrap_err(),CaptureContiguousProjectionError::Fragments);
    }
}
