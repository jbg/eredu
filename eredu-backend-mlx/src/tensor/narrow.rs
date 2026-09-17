//! Exact portable axis interval through the existing native Slice worker.
use super::*;
use eredu_nn::{TensorAxisRange,TensorAxisRangeError};
use std::mem::{size_of,size_of_val};

#[derive(Debug,thiserror::Error)]
#[error("native axis range coordinate capacity differs")]
struct CoordinateCapacity;
fn error(cause:impl std::error::Error+Send+Sync+'static)->Error {
    match safemlx::OriginalScopeObserver::try_current() {
        Ok(None)=>Error::backend_source(cause),
        _=>Error::backend_retained_source(cause),
    }
}
pub(super) fn execute(input:&MlxTensor,axis:usize,start:i32,end:i32,stream:&Stream)->Result<MlxTensor,Error>{
    let range=TensorAxisRange::new(input.shape(),axis,start,end).map_err(error)?;
    let rank=range.shape().len();
    // Four axes are the current exact CPU Slice source domain. This is an
    // inline storage choice, not a semantic rank limit; larger ordinary inputs
    // use only their actual input-derived coordinate population below.
    if rank<=4 {
        let mut starts=[0;4]; let mut ends=[0;4]; let strides=[1;4];
        ends[..rank].copy_from_slice(range.shape());
        starts[range.axis()]=range.start(); ends[range.axis()]=range.end();
        return input.as_array().try_slice(&starts[..rank],&ends[..rank],&strides[..rank],stream)
            .map(MlxTensor::from_array).map_err(error);
    }
    let mut starts=Vec::new(); let mut ends=Vec::new(); let mut strides=Vec::new();
    for coordinates in [&mut starts,&mut ends,&mut strides] {
        coordinates.try_reserve_exact(rank).map_err(error)?;
        if coordinates.capacity()!=rank {
            return Err(error(CoordinateCapacity));
        }
    }
    starts.resize(rank,0); ends.extend_from_slice(range.shape()); strides.resize(rank,1);
    starts[range.axis()]=range.start(); ends[range.axis()]=range.end();
    input.as_array().try_slice(&starts,&ends,&strides,stream).map(MlxTensor::from_array).map_err(error)
}

/// Fixed transports plus the exact three out-of-line coordinate vectors when
/// rank exceeds the inline source domain. This is only a layout; the ordinary
/// Slice/identity constructor, CPU/Metal worker and completion sources remain
/// separately required. Workspace StaticSlice may be invoked by either the
/// direct capture caller or this adapter, so its bound covers both callers.
pub(crate) fn control_bytes(rank:usize)->Option<usize>{
    let frames=[size_of::<TensorAxisRange<'_>>(),size_of::<Result<TensorAxisRange<'_>,TensorAxisRangeError>>(),
        size_of::<(&MlxTensor,usize,i32,i32,&Stream)>()*2,size_of::<[i32;4]>()*3,
        size_of::<[Vec<i32>;3]>(),size_of::<[&mut Vec<i32>;3]>(),
        size_of::<std::array::IntoIter<&mut Vec<i32>,3>>(),size_of::<usize>(),
        size_of::<Result<(),std::collections::TryReserveError>>(),size_of::<std::collections::TryReserveError>(),
        size_of::<Result<MlxTensor,Error>>()*2,size_of::<Result<Array,safemlx::error::Exception>>(),
        size_of::<(&Array,&[i32],&[i32],&[i32],&Stream)>(),
        size_of::<std::slice::Iter<'_,i32>>(),size_of::<[i32;4]>(),size_of::<Option<i32>>(),
        size_of::<safemlx::OriginalScopeObserver>(),
        size_of::<Result<Option<safemlx::OriginalScopeObserver>,safemlx::error::Exception>>(),
        safemlx::OriginalScopeObserver::control_bytes()?,
        Error::retained_source_control_bytes::<safemlx::error::Exception>()?,
        Error::retained_source_control_bytes::<TensorAxisRangeError>()?,
        Error::retained_source_control_bytes::<std::collections::TryReserveError>()?,
        Error::retained_source_control_bytes::<CoordinateCapacity>()?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?
        .checked_add(if rank>4{rank.checked_mul(3)?.checked_mul(size_of::<i32>())?}else{0})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore="requires a native CPU execution device"]
    fn native_axis_range_preserves_batched_values_empty_shapes_and_source_aliases(){
        let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
        for shape in [vec![2,5,3],vec![2,1,5,1,3]] {
            let axis=if shape.len()==3{1}else{2};
            let values:Vec<f32>=(0..30).map(|n|n as f32*0.25-2.0).collect();
            let input=MlxTensor::from_array(Array::from_slice(&values,&shape));
            for (start,end) in [(3,5),(4,5),(0,5),(5,5)] {
                let selected=input.narrow_axis(axis,start,end,&stream).unwrap();
                let range=TensorAxisRange::new(&shape,axis,start,end).unwrap();
                let ordinary=input.index(&range.indexes().collect::<Vec<_>>(),&stream).unwrap();
                assert_eq!(selected.shape(),ordinary.shape());
                let actual=selected.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap();
                assert_eq!(actual,ordinary.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap());
                let mut expected=Vec::new();
                for batch in 0..2 {for position in start..end {for width in 0..3 {
                    expected.push(values[batch*15+position as usize*3+width]);
                }}}
                assert_eq!(actual,expected);
                assert_eq!(input.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap(),values);
                drop(selected);drop(ordinary);
            }
            for (axis,start,end) in [(shape.len(),0,1),(axis,-1,1),(axis,3,2),(axis,0,6)] {
                assert!(input.narrow_axis(axis,start,end,&stream).is_err());
            }
        }
        assert_eq!(control_bytes(5).unwrap()-control_bytes(4).unwrap(),3*5*size_of::<i32>());
    }
}
