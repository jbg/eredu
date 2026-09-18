//! Unicode 17 standard-library lowercase semantics with an admitted output.
//! Regex property matching itself retains regex-syntax's Unicode 16 tables.
mod tables;
use crate::allocation::{AllocationError, Context};
use alloc::string::String;

fn contains(ranges: &[(char, char)], c: char) -> bool {
    ranges
        .binary_search_by(|&(start, end)| {
            if end < c {
                core::cmp::Ordering::Less
            } else if start > c {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}
fn chars(text: &str) -> impl Iterator<Item = char> + '_ {
    let mut previous_cased = false;
    text.char_indices().flat_map(move |(offset, c)| {
        let final_sigma = c == 'Σ'
            && previous_cased
            && !text[offset + c.len_utf8()..]
                .chars()
                .find(|&next| !contains(tables::IGNORABLE, next))
                .is_some_and(|next| contains(tables::CASED, next));
        if !contains(tables::IGNORABLE, c) {
            previous_cased = contains(tables::CASED, c);
        }
        if final_sigma {
            'ς'.to_lowercase()
        } else {
            c.to_lowercase()
        }
    })
}
pub(super) fn lowercase(text: &str, allocation: Context<'_>) -> Result<String, AllocationError> {
    let bytes = chars(text).try_fold(0usize, |n, c| {
        n.checked_add(c.len_utf8())
            .ok_or(AllocationError::SizeOverflow)
    })?;
    allocation.storage.reserve(bytes)?;
    let mut output = String::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| AllocationError::HostAllocation)?;
    for c in chars(text) {
        output.push(c);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::{Allocation, Unenforced};
    use alloc::format;
    use core::cell::Cell;
    #[test]
    fn every_scalar_and_sigma_context_matches_the_pinned_standard_library() {
        assert_eq!(char::UNICODE_VERSION, (17, 0, 0));
        let allocation = Context::new(&Unenforced);
        // Batches keep test memory fixed while independently checking complete output.
        let mut inputs: [String; 4] = core::array::from_fn(|_| String::new());
        for c in (0..=0x10ffff).filter_map(char::from_u32) {
            for (text, piece) in inputs.iter_mut().zip([
                format!("#{c}"),
                format!("#{c}Σ"),
                format!("#AΣ{c}"),
                format!("#AΣ{c}A"),
            ]) {
                text.push_str(&piece);
            }
            if c as u32 % 1024 == 0 {
                for text in &mut inputs {
                    assert_eq!(lowercase(text, allocation).unwrap(), text.to_lowercase());
                    text.clear();
                }
            }
        }
        for text in inputs {
            assert_eq!(lowercase(&text, allocation).unwrap(), text.to_lowercase());
        }
        for text in [
            "ΟΣ",
            "ΟΣ'Α",
            "ΟΣ'",
            "İΣ\u{345}Σ",
            "AΣ\u{301}\u{200d}A",
            "AΣ\u{301}\u{200d}",
        ] {
            assert_eq!(lowercase(text, allocation).unwrap(), text.to_lowercase());
        }
    }
    struct Funding {
        requests: Cell<usize>,
        bytes: Cell<usize>,
        refuse: bool,
    }
    impl Allocation for Funding {
        fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
            self.requests.set(self.requests.get() + 1);
            self.bytes.set(bytes);
            if self.refuse {
                Err(AllocationError::Refused)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn exact_output_request_precedes_every_allocation() {
        for input in ["", "Uppercase", "İΣ\u{301}"] {
            let expected = input.to_lowercase();
            for refuse in [false, true] {
                let funding = Funding {
                    requests: Cell::new(0),
                    bytes: Cell::new(0),
                    refuse,
                };
                let output = lowercase(input, Context::new(&funding));
                if refuse && !input.is_empty() {
                    assert_eq!(output, Err(AllocationError::Refused));
                } else {
                    assert_eq!(output.unwrap(), expected);
                }
                assert_eq!(funding.requests.get(), usize::from(!input.is_empty()));
                assert_eq!(funding.bytes.get(), expected.len());
            }
        }
    }
}
