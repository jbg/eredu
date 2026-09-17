//! Canonical ownership lookup shared by ordinary and paid sparse receipts.
use super::*;
#[derive(Debug)]
pub(in crate::capture::partition) enum RoutedOwnership {
    Ordinary(BTreeMap<usize,RoutedUnitCaptureOwnership>),
    Prepared(Vec<(usize,RoutedUnitCaptureOwnership)>),
}
impl From<BTreeMap<usize,RoutedUnitCaptureOwnership>> for RoutedOwnership {
    fn from(value:BTreeMap<usize,RoutedUnitCaptureOwnership>)->Self {Self::Ordinary(value)}
}
impl RoutedOwnership {
    pub(super) fn prepared(mut rows:Vec<(usize,RoutedUnitCaptureOwnership)>)->Result<Self,CaptureError> {
        // Fixed insertion controls: retain the paid Vec and allocate no sort scratch.
        for index in 1..rows.len() {
            let mut at=index;
            while at>0&&rows[at-1].0>rows[at].0 {rows.swap(at-1,at);at-=1;}
        }
        if rows.windows(2).any(|pair|pair[0].0==pair[1].0) {
            return Err(invalid("duplicate sparse producer"));
        }
        Ok(Self::Prepared(rows))
    }
    pub(in crate::capture::partition) fn len(&self)->usize {match self {Self::Ordinary(rows)=>rows.len(),Self::Prepared(rows)=>rows.len()}}
    pub(in crate::capture::partition) fn is_empty(&self)->bool {self.len()==0}
    pub(in crate::capture::partition) fn get(&self,rank:&usize)->Option<&RoutedUnitCaptureOwnership> {
        match self {Self::Ordinary(rows)=>rows.get(rank),Self::Prepared(rows)=>rows.binary_search_by_key(rank,|row|row.0).ok().map(|index|&rows[index].1)}
    }
    pub(in crate::capture::partition) fn iter(&self)->OwnershipIter<'_> {
        match self {Self::Ordinary(rows)=>OwnershipIter::Ordinary(rows.iter()),
            Self::Prepared(rows)=>OwnershipIter::Prepared(rows.iter())}
    }
    pub(super) fn control_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts=[size_of::<OwnershipIter<'_>>()*3,size_of::<Self>()*2,
            size_of::<Result<usize,usize>>(),size_of::<Option<&RoutedUnitCaptureOwnership>>(),
            size_of::<(&usize,&RoutedUnitCaptureOwnership)>(),size_of::<[usize;5]>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(in crate::capture::partition) fn values(&self)->impl Iterator<Item=&RoutedUnitCaptureOwnership> {
        self.iter().map(|(_,owned)|owned)
    }
}

pub(in crate::capture::partition) enum OwnershipIter<'a> {
    Ordinary(std::collections::btree_map::Iter<'a,usize,RoutedUnitCaptureOwnership>),
    Prepared(std::slice::Iter<'a,(usize,RoutedUnitCaptureOwnership)>),
}
impl<'a> Iterator for OwnershipIter<'a> {
    type Item=(&'a usize,&'a RoutedUnitCaptureOwnership);
    fn next(&mut self)->Option<Self::Item> {
        match self {Self::Ordinary(rows)=>rows.next(),Self::Prepared(rows)=>rows.next().map(|(rank,owned)|(rank,owned))}
    }
}
