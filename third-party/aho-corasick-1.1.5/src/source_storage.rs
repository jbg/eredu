//! Nonallocating census of actual retained source owners.
use alloc::{sync::Arc, vec::Vec};
use core::{alloc::Layout, mem, sync::atomic::AtomicUsize};
pub(crate) type Visitor<'a> = dyn FnMut(*const (), usize) -> bool + 'a;
pub(crate) fn arc<T: ?Sized>(
    value: &Arc<T>,
    visitor: &mut Visitor<'_>,
) -> bool {
    let (layout, offset) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::for_value(&**value))
        .expect("actual Arc layout");
    let bytes = layout.pad_to_align().size();
    let identity =
        Arc::as_ptr(value).cast::<u8>().wrapping_sub(offset).cast::<()>();
    visitor(identity, bytes)
}
pub(crate) fn vector<T>(value: &Vec<T>, visitor: &mut Visitor<'_>) -> bool {
    let bytes = value.capacity() * mem::size_of::<T>();
    bytes == 0 || visitor(value.as_ptr().cast::<()>(), bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AhoCorasick, AhoCorasickKind, MatchKind};
    use alloc::collections::BTreeMap;

    fn census(ac: &AhoCorasick, seen: &mut BTreeMap<usize, usize>) -> usize {
        let mut total = 0;
        ac.visit_source_storage(&mut |id, bytes| {
            assert!(bytes > 0);
            if let Some(previous) = seen.insert(id as usize, bytes) {
                assert_eq!(previous, bytes);
                false
            } else {
                total += bytes;
                true
            }
        });
        total
    }
    #[test]
    fn all_automata_share_actual_census_across_aliases_and_searches() {
        let patterns: Vec<_> = (0..131)
            .map(|index| alloc::format!("prefix-{index:03}"))
            .collect();
        for kind in [
            AhoCorasickKind::NoncontiguousNFA,
            AhoCorasickKind::ContiguousNFA,
            AhoCorasickKind::DFA,
        ] {
            for prefilter in [false, true] {
                let ac = AhoCorasick::builder()
                    .kind(Some(kind))
                    .match_kind(MatchKind::LeftmostFirst)
                    .prefilter(prefilter)
                    .build(&patterns)
                    .unwrap();
                let mut before = BTreeMap::new();
                let total = census(&ac, &mut before);
                assert!(total > 0);
                assert_eq!(census(&ac.clone(), &mut before), 0);
                assert_eq!(
                    ac.find("prefix-130").unwrap().pattern().as_usize(),
                    130
                );
                let mut after = BTreeMap::new();
                assert_eq!(census(&ac, &mut after), total);
                assert_eq!(before, after);
                let mut calls = 0;
                ac.visit_source_storage(&mut |_, _| {
                    calls += 1;
                    false
                });
                assert_eq!(
                    calls, 1,
                    "declining the shared root skips its complete backing"
                );
            }
        }
    }
    #[test]
    fn vector_census_counts_spare_capacity() {
        let mut values = Vec::<u64>::with_capacity(17);
        values.push(5);
        let mut records = Vec::new();
        vector(&values, &mut |id, bytes| {
            records.push((id, bytes));
            true
        });
        assert_eq!(records, [(values.as_ptr().cast(), values.capacity() * 8)]);
    }
}
