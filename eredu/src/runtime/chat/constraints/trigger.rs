//! Ordinary and prepared semantic controllers use the same borrowed byte worker.
pub(super) use eredu_core::speculative::byte_trigger::{TriggerPrefix, find, next_prefix};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_trigger_matches_whole_split_overlap_and_binary_activation() {
        for trigger in [b"abab".as_slice(), b"\0\xff\0", "é🙂".as_bytes()] {
            let mut input = b"prefix".to_vec();
            input.extend_from_slice(trigger);
            input.extend_from_slice(b"tail");
            for split in 0..=input.len() {
                let mut prefix = TriggerPrefix::default();
                let first = find(&[], &input[..split], trigger);
                prefix.advance(&input[..split], trigger);
                let second = find(prefix.bytes(trigger), &input[split..], trigger);
                assert!(first.is_some() || second.is_some());
                if let Some(found) = first.or(second) {
                    let actual: Vec<_> = found.prefix.iter().chain(found.tail).copied().collect();
                    assert!(actual.starts_with(trigger));
                }
            }
        }
        let found = find(b"ab", b"abtail", b"abab").unwrap();
        assert_eq!(found.prefix, b"abab");
        assert_eq!(found.tail, b"tail");
        assert!(!found.starts_at_token_boundary);
        assert!(
            find(&[], b"ababtail", b"abab")
                .unwrap()
                .starts_at_token_boundary
        );
        assert!(find(b"ab", b"ax", b"abab").is_none());
        assert!(find(&[], b"anything", &[]).is_none());
    }

    #[test]
    fn finite_prefix_matches_concatenation_oracle_for_every_chunk_boundary() {
        for trigger in [b"abab".as_slice(), b"aaaa", b"\0\xff\0", "é🙂".as_bytes()] {
            for input in [
                b"abababaabab".as_slice(),
                b"aaaaaabaaaa",
                b"\0\xff\0\xff\0",
                "xé🙂é🙂".as_bytes(),
            ] {
                for split in 0..=input.len() {
                    let mut prefix = TriggerPrefix::default();
                    let mut seen = Vec::new();
                    for chunk in [&input[..split], &input[split..]] {
                        seen.extend_from_slice(chunk);
                        let expected = (0..trigger.len())
                            .rev()
                            .find(|&keep| seen.ends_with(&trigger[..keep]))
                            .unwrap();
                        prefix.advance(chunk, trigger);
                        assert_eq!(prefix.bytes(trigger), &trigger[..expected]);
                    }
                }
            }
        }
    }
}
