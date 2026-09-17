//! Borrowed input handles and paid destinations for the shared uniqueness worker.
use super::{workspace::{self, Error}, ValidationContext};
use crate::{Array, Json, Node};
use hashbrown::HashTable;
use jsonschema_value::{cmp, unique::{self, UniqueStep}};
use std::{alloc::Layout, hash::{BuildHasher, Hasher}, mem::{size_of, size_of_val}};

type Entry = (u64, usize);
impl ValidationContext<'_> {
    pub(crate) fn unique_items<'a, F: Json>(
        &mut self, array: &<F::Node<'a> as Node<'a,F>>::Array,
    ) -> bool {
        let parts = [
            size_of::<(&mut Self, &<F::Node<'a> as Node<'a,F>>::Array)>(),
            size_of::<Vec<F::Node<'a>>>(), size_of::<F::Node<'a>>(),
            size_of::<<<F::Node<'a> as Node<'a,F>>::Array as Array<'a,F>>::ElementsIter>(),
            size_of::<HashTable<Entry>>(), size_of::<Entry>(),
            size_of::<ahash::RandomState>(), size_of::<ahash::AHasher>(),
            size_of::<UniqueStep>(), size_of::<(usize,usize,bool)>(),
            size_of::<Result<Layout,std::alloc::LayoutError>>(),
            size_of::<Result<(),std::collections::TryReserveError>>(),
            size_of::<Error>(),
            size_of::<(&mut Self,&mut HashTable<Entry>,&Vec<F::Node<'a>>)>()];
        if !self.workspace.reserve(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)) { return false; }
        let length = array.len();
        let Ok(layout) = Layout::array::<F::Node<'a>>(length) else { return self.workspace.refuse(Error::Overflow); };
        if !self.workspace.reserve(Some(layout.size())) { return false; }
        let mut items = Vec::new();
        if let Err(cause) = items.try_reserve_exact(length) { return self.workspace.refuse(Error::Allocation(cause)); }
        if size_of::<F::Node<'a>>() != 0 && items.capacity()>length { return self.workspace.refuse(Error::Capacity); }
        for node in array.elements() {
            if items.len()==length { return self.workspace.refuse(Error::Capacity); }
            items.push(node);
        }
        if items.len()!=length { return self.workspace.refuse(Error::Capacity); }
        let seed = self.workspace.hasher();
        let mut table = HashTable::<Entry>::new();
        unique::is_unique_with(length, &mut |step| {
            if self.workspace.failed() { return false; }
            let controls = size_of::<(UniqueStep, &mut Self, &mut HashTable<Entry>, &Vec<F::Node<'a>>, bool)>();
            if !self.workspace.reserve(Some(controls)) { return false; }
            match step {
                UniqueStep::Controls(bytes) => self.workspace.reserve(bytes),
                UniqueStep::Compare(a,b) => {
                    let equal = cmp::equal_nodes_with::<F,_>(&items[a],&items[b], &mut |charge| self.equality_charge(charge));
                    !self.workspace.failed() && !equal
                }
                UniqueStep::HashTable(size) => self.unique_reserve(&mut table, size),
                UniqueStep::Insert(index) => {
                    let mut hasher = seed.build_hasher();
                    let input_controls = self.workspace.input_controls();
                    if !unique::hash_node_with::<F,_,_,_>(&items[index], &mut hasher,
                        &mut || seed.build_hasher(),
                        &mut |bytes| self.workspace.reserve(bytes.and_then(|n| n.checked_add(input_controls))))
                    { return false; }
                    self.unique_insert::<F>(&mut table, hasher.finish(), index, &items)
                }
            }
        })
    }

    fn unique_reserve(&mut self, table: &mut HashTable<Entry>, length: usize) -> bool {
        let parts = [
            size_of::<(&mut Self,&mut HashTable<Entry>,usize)>(),
            size_of::<Result<Option<Layout>,hashbrown::TryReserveError>>(),
            size_of::<Result<(),hashbrown::TryReserveError>>(), size_of::<Error>(),
            size_of::<(&Entry,u64)>()];
        if !self.workspace.reserve(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)) { return false; }
        let request = match table.try_reserve_layout(length) {
            Ok(request) => request,
            Err(cause) => return self.workspace.refuse(Error::Table(cause)),
        };
        if let Some(layout) = request {
            if !self.workspace.reserve(Some(layout.size())) { return false; }
        }
        if let Err(cause) = table.try_reserve(length, |entry| entry.0) {
            return self.workspace.refuse(Error::Table(cause));
        }
        true
    }

    fn unique_insert<'a,F: Json>(&mut self, table: &mut HashTable<Entry>, hash: u64, index: usize, items: &[F::Node<'a>]) -> bool {
        let parts = [size_of::<(&mut Self,&mut HashTable<Entry>,u64,usize,&[F::Node<'a>])>(),
            size_of::<Entry>(), size_of::<Option<&Entry>>(), size_of::<bool>(),
            size_of::<hashbrown::hash_table::OccupiedEntry<'_,Entry>>(),
            size_of::<(&Entry,&mut Self,&[F::Node<'a>],usize)>()];
        if !self.workspace.reserve(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)) { return false; }
        // The exact closure is queried before any table probe. Pay the query
        // result before calling the existing borrowed find/equality worker.
        let lookup = {
            let equality = |entry: &Entry| cmp::equal_nodes_with::<F,_>(&items[entry.1],&items[index],
                &mut |charge| self.equality_charge(charge));
            table.lookup_control_bytes(&equality)
        };
        if !self.workspace.reserve(lookup) { return false; }
        let duplicate = table.find(hash, |entry| cmp::equal_nodes_with::<F,_>(&items[entry.1], &items[index],
            &mut |charge| self.equality_charge(charge))).is_some();
        if self.workspace.failed() || duplicate { return false; }
        let hasher = |entry: &Entry| entry.0;
        let Some(controls) = table.reserved_insert_control_bytes(&hasher) else {
            return self.workspace.refuse(Error::Capacity);
        };
        if !self.workspace.reserve(Some(controls)) { return false; }
        table.insert_unique(hash, (hash,index), hasher);
        true
    }
}
