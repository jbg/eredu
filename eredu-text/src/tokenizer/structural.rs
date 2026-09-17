//! Shared structural-token validation over exact tokenizer facts and encoding.

/// Operations required by structural-token validation. Encoding retains its
/// actual source/operation owner; callers choose ordinary or admitted storage.
pub trait StructuralTokenSource {
    /// Completed encoding, including any immutable source custody.
    type Encoded;
    /// Actual encoder failure, including its failed storage prefix.
    type Error;
    /// Exact added-token membership, distinct from model vocabulary membership.
    fn added_id(&self, spelling: &str) -> Option<u32>;
    /// Canonical forward lookup from the same tokenizer.
    fn token_id(&self, spelling: &str) -> Option<u32>;
    /// Whether the actual reverse vocabulary spells this exact token.
    fn roundtrips(&self, id: u32, spelling: &str) -> bool;
    /// Encodes without special-token insertion through the selected worker.
    fn encode(&mut self, spelling: &str) -> Result<Self::Encoded, Self::Error>;
    /// Borrows the completed encoding without cloning IDs or their owner.
    fn encoded_ids(encoded: &Self::Encoded) -> &[u32];
}

/// Fixed validation failure. Encoding failures and non-atomic encodings keep
/// their original payload; no formatted diagnostic or detached ID copy is made.
#[derive(Debug)]
pub enum StructuralTokenFailure<E, T> {
    /// The caller did not supply exactly one result slot per spelling.
    Destination,
    /// A required spelling is empty.
    Empty { index: usize },
    /// A spelling repeats an earlier declaration.
    Repeated { index: usize },
    /// The spelling is not an added token.
    MissingAdded { index: usize },
    /// Added and canonical forward IDs differ.
    Forward { index: usize, id: u32 },
    /// The actual reverse vocabulary has another spelling.
    Reverse { index: usize, id: u32 },
    /// The shared encoder failed, retaining its actual failure custody.
    Encoding { index: usize, cause: E },
    /// Encoding did not produce exactly the registered added ID.
    NonAtomic { index: usize, id: u32, encoded: T },
    /// Distinct spellings resolve to the same ID.
    Ambiguous {
        previous: usize,
        index: usize,
        id: u32,
    },
}

/// Validate in ordinary declaration order into an exact caller-owned result.
/// Duplicate checks borrow the prior spelling/result prefix rather than create
/// maps. A failure leaves that accepted prefix intact and grants no profile.
pub fn resolve_structural_with<S: StructuralTokenSource, T: AsRef<str>>(
    source: &mut S,
    spellings: &[T],
    resolved: &mut [u32],
) -> Result<(), StructuralTokenFailure<S::Error, S::Encoded>> {
    use StructuralTokenFailure as F;
    if spellings.len() != resolved.len() {
        return Err(F::Destination);
    }
    for (index, spelling) in spellings.iter().enumerate() {
        let spelling = spelling.as_ref();
        if spelling.is_empty() {
            return Err(F::Empty { index });
        }
        if spellings[..index]
            .iter()
            .any(|prior| prior.as_ref() == spelling)
        {
            return Err(F::Repeated { index });
        }
        let id = source.added_id(spelling).ok_or(F::MissingAdded { index })?;
        if source.token_id(spelling) != Some(id) {
            return Err(F::Forward { index, id });
        }
        if !source.roundtrips(id, spelling) {
            return Err(F::Reverse { index, id });
        }
        let encoded = source
            .encode(spelling)
            .map_err(|cause| F::Encoding { index, cause })?;
        if S::encoded_ids(&encoded) != [id] {
            return Err(F::NonAtomic { index, id, encoded });
        }
        if let Some(previous) = resolved[..index].iter().position(|prior| *prior == id) {
            return Err(F::Ambiguous {
                previous,
                index,
                id,
            });
        }
        resolved[index] = id;
    }
    Ok(())
}

/// Exact shared fixed frames; source operations and the caller's result array
/// are separately charged by their actual producers.
pub fn structural_control_bytes<S: StructuralTokenSource, T>() -> Option<usize> {
    use std::mem::size_of;
    let parts = [
        size_of::<&mut S>(),
        size_of::<&[T]>(),
        size_of::<&mut [u32]>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, T>>>(),
        size_of::<std::slice::Iter<'_, T>>(),
        size_of::<std::slice::Iter<'_, u32>>(),
        size_of::<(usize, &T, &str, u32)>(),
        size_of::<Option<usize>>(),
        size_of::<Option<u32>>(),
        size_of::<S::Encoded>(),
        size_of::<Result<S::Encoded, S::Error>>(),
        size_of::<StructuralTokenFailure<S::Error, S::Encoded>>(),
        size_of::<Result<(), StructuralTokenFailure<S::Error, S::Encoded>>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}

/// Ordinary adapter; admission-aware callers provide their own exact source.
pub struct OrdinaryStructuralTokens<'a>(pub &'a super::Tokenizer);
impl StructuralTokenSource for OrdinaryStructuralTokens<'_> {
    type Encoded = tokenizers::Encoding;
    type Error = tokenizers::Error;
    fn added_id(&self, spelling: &str) -> Option<u32> {
        self.0
            .get_added_vocabulary()
            .get_vocab()
            .get(spelling)
            .copied()
    }
    fn token_id(&self, spelling: &str) -> Option<u32> {
        self.0.token_to_id(spelling)
    }
    fn roundtrips(&self, id: u32, spelling: &str) -> bool {
        self.0.id_to_token(id).as_deref() == Some(spelling)
    }
    fn encode(&mut self, spelling: &str) -> Result<Self::Encoded, Self::Error> {
        self.0.encode(spelling, false)
    }
    fn encoded_ids(encoded: &Self::Encoded) -> &[u32] {
        encoded.get_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    struct Encoded {
        ids: Vec<u32>,
        _owner: Rc<()>,
    }
    struct Source {
        owner: Rc<()>,
        calls: usize,
    }
    impl StructuralTokenSource for Source {
        type Encoded = Encoded;
        type Error = ();
        fn added_id(&self, spelling: &str) -> Option<u32> {
            match spelling {
                "a" => Some(1),
                "b" => Some(2),
                _ => None,
            }
        }
        fn token_id(&self, spelling: &str) -> Option<u32> {
            self.added_id(spelling)
        }
        fn roundtrips(&self, id: u32, spelling: &str) -> bool {
            self.added_id(spelling) == Some(id)
        }
        fn encode(&mut self, spelling: &str) -> Result<Encoded, ()> {
            self.calls += 1;
            Ok(Encoded {
                ids: if spelling == "a" { vec![1] } else { vec![2, 3] },
                _owner: self.owner.clone(),
            })
        }
        fn encoded_ids(encoded: &Encoded) -> &[u32] {
            &encoded.ids
        }
    }
    #[test]
    fn exact_prefix_and_non_atomic_failure_keep_encoding_owner_until_retirement() {
        let owner = Rc::new(());
        let mut source = Source {
            owner: owner.clone(),
            calls: 0,
        };
        let mut output = [99; 2];
        let failure = resolve_structural_with(&mut source, &["a", "b"], &mut output);
        assert!(
            matches!(&failure, Err(StructuralTokenFailure::NonAtomic { index: 1, id: 2, encoded }) if encoded.ids == [2, 3])
        );
        assert_eq!(output, [1, 99]);
        assert_eq!(source.calls, 2);
        drop(source);
        assert_eq!(Rc::strong_count(&owner), 2);
        drop(failure);
        assert_eq!(Rc::strong_count(&owner), 1);
        let mut source = Source { owner, calls: 0 };
        let repeated = resolve_structural_with(&mut source, &["a", "a"], &mut output);
        assert!(matches!(
            repeated,
            Err(StructuralTokenFailure::Repeated { index: 1 })
        ));
        assert_eq!(source.calls, 1);
    }
}
