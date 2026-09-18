use super::*;
use eredu_core::HostMetadataAccount;
use eredu_runtime::working_memory::WorkingMemoryPool;
use eredu_text::tokenizer_storage::TokenizerPlan;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};

#[derive(Debug)]
struct Account {
    refuse: Arc<AtomicBool>,
    retired: Arc<AtomicBool>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {
            Err(HostMetadataFundingError::Capacity {
                required: bytes.try_into().unwrap(),
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}
fn funding() -> (HostMetadataFunding, Arc<AtomicBool>, Arc<AtomicBool>) {
    let refuse = Arc::new(AtomicBool::new(false));
    let retired = Arc::new(AtomicBool::new(false));
    (
        HostMetadataFunding::new(Account {
            refuse: refuse.clone(),
            retired: retired.clone(),
        })
        .unwrap(),
        refuse,
        retired,
    )
}
fn tokenizer() -> tokenizers::Tokenizer {
    let model = BPE::builder()
        .vocab_and_merges([("a".to_owned(), 0), ("b".to_owned(), 5)], Vec::new())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<eos>", true).normalized(false),
            AddedToken::from("<|end|>", true).normalized(false),
            AddedToken::from("<unk>", true).normalized(false),
        ])
        .unwrap();
    tokenizer
        .add_tokens([AddedToken::from("<pad>", false).normalized(false)])
        .unwrap();
    tokenizer
}
fn source(pool: &WorkingMemoryPool, raw: &tokenizers::Tokenizer) -> OriginalTokenizer {
    let json = raw.to_string(false).unwrap();
    pool.compile_tokenizer(TokenizerPlan::prepare_json(json.as_bytes()).unwrap())
        .unwrap()
}

#[test]
fn original_compiler_preserves_sparse_vocabulary_special_metadata_and_exact_eos() {
    let raw = tokenizer();
    let eos = [
        raw.token_to_id("<eos>").unwrap(),
        raw.token_to_id("<|end|>").unwrap(),
    ];
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let (funding, _, retired) = funding();
    let compiler =
        ConstraintCompiler::from_original_tokenizer(source(&pool, &raw), &eos, &funding).unwrap();
    let ordinary = crate::runtime::chat::tokenizer_env::from_tokenizer(
        &eredu_text::tokenizer::Tokenizer::from_tokenizer(raw),
        &eos,
    )
    .unwrap();
    assert_eq!(compiler.trie().info(), ordinary.tok_trie().info());
    assert_eq!(compiler.trie().eos_tokens(), &eos);
    assert_eq!(compiler.trie().info().tok_pad, None);
    for id in 0..compiler.trie().vocab_size() {
        assert_eq!(
            compiler.trie().token(id as u32),
            ordinary.tok_trie().token(id as u32)
        );
    }
    let grammar = compiler
        .compile_grammar(llguidance::api::TopLevelGrammar::from_lark(
            "start: \"a\"".into(),
        ))
        .unwrap();
    drop((funding, compiler));
    assert!(!retired.load(Ordering::SeqCst));
    drop(grammar);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn empty_configured_eos_does_not_reclassify_id_zero_or_detected_specials() {
    let raw = tokenizer();
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let (funding, _, _) = funding();
    let compiler =
        ConstraintCompiler::from_original_tokenizer(source(&pool, &raw), &[], &funding).unwrap();
    let ordinary = crate::runtime::chat::tokenizer_env::from_tokenizer(
        &eredu_text::tokenizer::Tokenizer::from_tokenizer(raw),
        &[],
    )
    .unwrap();
    for trie in [compiler.trie(), ordinary.tok_trie()] {
        assert_eq!(trie.info().tok_eos, INVALID_TOKEN);
        assert!(trie.eos_tokens().is_empty());
        assert_eq!(trie.token(0), b"a");
        assert!(!trie.eos_token_set().is_allowed(0));
    }
    assert_eq!(compiler.trie().info(), ordinary.tok_trie().info());
}

#[test]
fn rejected_metadata_and_unmapped_eos_retain_original_source_and_payer() {
    for refuse_metadata in [false, true] {
        let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
        let source = source(&pool, &tokenizer());
        let original_bytes = pool.used_bytes().unwrap();
        let (funding, refuse, retired) = funding();
        refuse.store(refuse_metadata, Ordering::SeqCst);
        let error = ConstraintCompiler::from_original_tokenizer(source, &[u32::MAX], &funding)
            .err()
            .unwrap();
        if refuse_metadata {
            assert!(matches!(
                error.cause,
                CompilerSourceCause::Metadata(HostMetadataFundingError::Capacity { .. })
            ));
        } else {
            assert!(matches!(error.cause, CompilerSourceCause::Eos(u32::MAX)));
        }
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        assert_eq!(pool.used_bytes().unwrap(), original_bytes);
        drop(error);
        assert!(retired.load(Ordering::SeqCst));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn declaration_shell_refusal_preserves_compiler_funding_after_source_retirement() {
    let raw = tokenizer();
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let (funding, refuse, retired) = funding();
    let compiler = ConstraintCompiler::from_original_tokenizer(source(&pool, &raw), &[], &funding).unwrap();
    let grammar = compiler.compile_grammar(llguidance::api::TopLevelGrammar::from_lark("start: \"a\"".into())).unwrap();
    refuse.store(true, Ordering::SeqCst);
    let error = super::super::declaration::PendingGrammarDeclaration::from_compiled(
        grammar, &compiler._authority, &compiler.allocation_funding).err().unwrap();
    drop((compiler, funding));
    assert!(!retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut found = false;
    while let Some(error) = cause {
        found |= matches!(error.downcast_ref::<HostMetadataFundingError>(),
            Some(HostMetadataFundingError::Capacity { available: 0, .. }));
        cause = error.source();
    }
    assert!(found);
    drop(error);
    assert!(retired.load(Ordering::SeqCst));
}
