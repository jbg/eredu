//! Dense ascending integer set with visible prospective word storage.
use alloc::vec::Vec;
use regex_syntax::allocation::{Allocation, AllocationError, Allocator, Unenforced};

/// Dense capture/group membership used by the parser and analyzer.
#[derive(Clone, Debug, Default)]
pub struct BitSet {
    words: Vec<u32>,
}
impl BitSet {
    /// Creates an empty set without allocating.
    pub fn new() -> Self {
        Self::default()
    }
    /// Inserts a member through the ordinary allocation policy.
    pub fn insert(&mut self, member: usize) -> bool {
        self.insert_with_allocations(member, &Unenforced)
            .expect("integer set allocation")
    }
    /// Funds a complete replacement word vector before growing or changing membership.
    pub fn insert_with_allocations(
        &mut self,
        member: usize,
        policy: &dyn Allocation,
    ) -> Result<bool, AllocationError> {
        let word = member / 32;
        let required = word.checked_add(1).ok_or(AllocationError::SizeOverflow)?;
        Allocator::new(policy).grow(&mut self.words, required)?;
        if self.words.len() < required {
            self.words.resize(required, 0);
        }
        let mask = 1 << (member % 32);
        let new = self.words[word] & mask == 0;
        self.words[word] |= mask;
        Ok(new)
    }
    /// Removes a member without allocating.
    pub fn remove(&mut self, member: usize) -> bool {
        let Some(word) = self.words.get_mut(member / 32) else {
            return false;
        };
        let mask = 1 << (member % 32);
        let existed = *word & mask != 0;
        *word &= !mask;
        existed
    }
    /// Checks membership without allocating.
    pub fn contains(&self, member: usize) -> bool {
        self.words
            .get(member / 32)
            .is_some_and(|word| word & (1 << (member % 32)) != 0)
    }
    /// Visits members in ascending order without temporary storage.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(index, mut word)| {
                core::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(index * 32 + bit)
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use core::cell::Cell;
    struct Funding {
        calls: Cell<usize>,
        refuse: usize,
    }
    impl Allocation for Funding {
        fn reserve(&self, _: usize) -> Result<(), AllocationError> {
            let at = self.calls.get();
            self.calls.set(at + 1);
            if at == self.refuse {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn membership_and_ascending_iteration_match_independent_tree_set() {
        let mut actual = BitSet::new();
        let mut expected = BTreeSet::new();
        let mut seed = 0x9234abcdu32;
        for _ in 0..8192 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let member = (seed as usize) % 2053;
            if seed & 1 == 0 {
                assert_eq!(actual.insert(member), expected.insert(member));
            } else {
                assert_eq!(actual.remove(member), expected.remove(&member));
            }
            assert_eq!(actual.contains(member), expected.contains(&member));
        }
        assert_eq!(
            actual.iter().collect::<Vec<_>>(),
            expected.into_iter().collect::<Vec<_>>()
        );
    }
    #[test]
    fn every_word_growth_refusal_preserves_prior_membership() {
        let members = [0, 31, 32, 63, 64, 127, 128, 511, 1024];
        let funding = Funding {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        let mut set = BitSet::new();
        for member in members {
            set.insert_with_allocations(member, &funding).unwrap();
        }
        let calls = funding.calls.get();
        assert!(calls > 1);
        for refuse in 0..calls {
            let funding = Funding {
                calls: Cell::new(0),
                refuse,
            };
            let mut set = BitSet::new();
            let mut prior = Vec::new();
            for member in members {
                if set.insert_with_allocations(member, &funding).is_err() {
                    assert_eq!(set.iter().collect::<Vec<_>>(), prior);
                    assert_eq!(funding.calls.get(), refuse + 1);
                    break;
                }
                prior.push(member);
            }
        }
    }
}
