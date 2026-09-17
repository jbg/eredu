use super::*;
use eredu_core::SharedStorageDomain;
use eredu_runtime::working_memory::WorkingMemoryPool;
use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo};
use std::sync::Arc;

fn plan() -> VocabularyPlan {
    let words = vec![
        b"hello".to_vec(),
        Vec::new(),
        "é🙂".as_bytes().to_vec(),
        b"\xff<|tool|>".to_vec(),
        Vec::new(),
        b"\xff".to_vec(),
    ];
    let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 0), &words);
    VocabularyPlan::new(Arc::new(ApproximateTokEnv::new(trie))).unwrap()
}

#[test]
fn packed_vocabulary_preserves_sparse_multibyte_and_special_token_slices() {
    let plan = plan();
    let expected: &[&[u8]] = &[b"hello", b"", "é🙂".as_bytes(), b"<|tool|>", b"", b""];
    assert_eq!(plan.layout.tokens, expected.len());
    assert_eq!(plan.layout.table_bytes, (expected.len() + 1) * 8);
    assert_eq!(plan.layout.max_token_bytes, 8);
    let logical = plan.layout.table_bytes + expected.iter().map(|bytes| bytes.len()).sum::<usize>();
    assert_eq!(plan.layout.total_bytes, logical);
    let vocabulary = plan.unregistered();
    assert_eq!(vocabulary.iter().collect::<Vec<_>>(), expected);
    assert_eq!(vocabulary.get(expected.len()), None);
    assert_eq!(vocabulary.get(usize::MAX), None);
    assert_eq!(vocabulary.bytes.as_ref().len(), logical);
    assert_eq!(
        u64::from_le_bytes(vocabulary.bytes.as_ref()[..8].try_into().unwrap()),
        ((expected.len() + 1) * 8) as u64
    );
    let final_offset = expected.len() * 8;
    assert_eq!(
        u64::from_le_bytes(
            vocabulary.bytes.as_ref()[final_offset..final_offset + 8]
                .try_into()
                .unwrap()
        ),
        logical as u64
    );
}

#[test]
fn spare_packed_capacity_and_preexisting_clones_retain_one_charge_until_last_alias() {
    let plan = plan();
    let mut packed = plan.pack();
    packed.reserve_exact(97);
    let capacity = packed.capacity() as u64;
    let pointer = packed.as_ptr();
    assert!(packed.capacity() > plan.layout.total_bytes);
    let vocabulary = SharedTokenVocabulary {
        bytes: SharedControllerBytes::new(packed),
        layout: plan.layout,
    };
    let alias = vocabulary.clone();
    let independent = plan.unregistered();
    assert!(alias.bytes.same_storage(&vocabulary.bytes));
    assert!(!independent.bytes.same_storage(&vocabulary.bytes));
    assert_eq!(alias.bytes.as_ref().as_ptr(), pointer);
    assert_eq!(alias.bytes.capacity_bytes(), Some(capacity));
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let domain = SharedStorageDomain::default();
    let identity = vocabulary.bytes.identity().clone();
    vocabulary
        .bytes
        .try_attach(&domain, || {
            pool.register_storage([(identity.clone(), capacity)])
                .map(|registration| Box::new(registration) as Box<dyn Send + Sync>)
        })
        .unwrap();
    drop(vocabulary);
    assert_eq!(pool.used_bytes().unwrap(), capacity);
    assert_eq!(alias.get(2), Some("é🙂".as_bytes()));
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // Metadata and independent equal contents must not keep the old charge.
    assert_ne!(&identity, independent.bytes.identity());
    drop((identity, independent));
}
