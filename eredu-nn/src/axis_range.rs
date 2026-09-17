//! Shared half-open, rank-preserving axis selection geometry.
use crate::Index;

/// A nonnegative axis interval does not fit its actual source geometry.
#[derive(Clone,Copy,Debug,PartialEq,Eq,thiserror::Error)]
#[error("tensor axis range is outside its source geometry")]
pub struct TensorAxisRangeError;

/// Validated nonnegative interval along one actual tensor axis. This describes
/// geometry only and grants no native storage, view or allocation authority.
#[derive(Clone,Copy,Debug)]
pub struct TensorAxisRange<'a> {
    shape: &'a [i32], axis:usize, start:i32, end:i32,
}
impl<'a> TensorAxisRange<'a> {
    /// Validates the exact shape, axis and half-open interval. Empty intervals
    /// are allowed; negative indexes and axis removal belong to `Tensor::index`.
    pub fn new(shape:&'a [i32],axis:usize,start:i32,end:i32)->Result<Self,TensorAxisRangeError>{
        if shape.iter().any(|&n|n<0) || shape.get(axis).is_none_or(|&n|start<0||end<start||end>n){
            return Err(TensorAxisRangeError);
        }
        Ok(Self{shape,axis,start,end})
    }
    /// Original rank and extents, borrowed from the actual source.
    pub fn shape(self)->&'a [i32]{self.shape}
    /// Axis retained by the view.
    pub fn axis(self)->usize{self.axis}
    /// Inclusive nonnegative start.
    pub fn start(self)->i32{self.start}
    /// Exclusive nonnegative end.
    pub fn end(self)->i32{self.end}
    /// Existing portable index semantics for a backend without a direct slice
    /// realization. This iterator itself allocates and retains nothing.
    pub fn indexes(self)->impl ExactSizeIterator<Item=Index>+'a {
        (0..self.shape.len()).map(move|axis|if axis==self.axis{Index::Range(self.start,self.end)}else{Index::Full})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn axis_range_retains_batch_and_rank_and_preserves_empty_intervals(){
        let source=[2,5,3];
        let selected=TensorAxisRange::new(&source,1,3,5).unwrap();
        assert_eq!(selected.shape(),source);
        assert_eq!(selected.indexes().collect::<Vec<_>>(),[Index::Full,Index::Range(3,5),Index::Full]);
        assert_eq!((selected.start(),selected.end(),selected.axis()),(3,5,1));
        assert!(TensorAxisRange::new(&source,1,5,5).is_ok());
        for (axis,start,end) in [(3,0,1),(1,-1,2),(1,3,2),(1,0,6)]{
            assert!(TensorAxisRange::new(&source,axis,start,end).is_err());
        }
        assert!(TensorAxisRange::new(&[],0,0,0).is_err());
        assert!(TensorAxisRange::new(&[2,-1,3],0,0,1).is_err());
    }
}
