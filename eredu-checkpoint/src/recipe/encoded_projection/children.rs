//! One preallocated child-owner and chunk array for a recursive join.
use super::{Mapping, RecipeError, overflow};
use std::{
    alloc::Layout, borrow::Borrow, collections::TryReserveError, marker::PhantomData, mem::size_of,
};

/// Checked storage for all children of one encoded join. No child is constructed
/// and no allocation is made during planning.
pub struct EncodedRecipeChildrenPlan<M> {
    count: usize,
    layout: Layout,
    marker: PhantomData<fn() -> M>,
}
impl<M> EncodedRecipeChildrenPlan<M> {
    pub(super) fn new(count: usize) -> Result<Self, RecipeError> {
        Ok(Self {
            count,
            layout: Layout::array::<(M, usize)>(count).map_err(|_| overflow())?,
            marker: PhantomData,
        })
    }
    /// Requested backing plus fixed constructor/result controls. Child mapping
    /// storage, allocator bookkeeping and machine stack are separate.
    pub fn required_bytes<C>(&self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<EncodedRecipeChildren<M, C>>(),
            size_of::<Result<EncodedRecipeChildren<M, C>, TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(self.layout.size(), usize::checked_add)
    }
    /// Reserve once before accepting any children. A refusal has no child prefix;
    /// the caller retains admission for the returned allocator error.
    pub fn construct<C>(self, custody: C) -> Result<EncodedRecipeChildren<M, C>, TryReserveError> {
        let mut entries = Vec::new();
        entries.try_reserve_exact(self.count)?;
        Ok(EncodedRecipeChildren {
            entries,
            count: self.count,
            custody,
        })
    }
}

/// Actual child owners and row widths. Child storage retires before custody;
/// only the checkpoint compiler can populate the preallocated array.
#[derive(Debug)]
pub struct EncodedRecipeChildren<M, C> {
    entries: Vec<(M, usize)>,
    count: usize,
    custody: C,
}
impl<M, C> EncodedRecipeChildren<M, C> {
    pub(super) fn push(&mut self, mapping: M, chunk: usize) -> Result<(), RecipeError> {
        if self.entries.len() == self.count {
            return Err(overflow());
        }
        self.entries.push((mapping, chunk));
        Ok(())
    }
    /// Move the existing array together with an additional owner. No admission,
    /// allocation or child mutation occurs during this ownership transfer.
    pub fn with_custody<D>(self, custody: D) -> EncodedRecipeChildren<M, (C, D)> {
        EncodedRecipeChildren {
            entries: self.entries,
            count: self.count,
            custody: (self.custody, custody),
        }
    }
}

pub(super) trait ChildMappings {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (&Mapping, usize);
}
impl<M: Borrow<Mapping>, C> ChildMappings for EncodedRecipeChildren<M, C> {
    fn len(&self) -> usize {
        self.entries.len()
    }
    fn get(&self, index: usize) -> (&Mapping, usize) {
        let (mapping, chunk) = &self.entries[index];
        (mapping.borrow(), *chunk)
    }
}
