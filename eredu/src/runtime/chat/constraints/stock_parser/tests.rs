use super::*;
use eredu_core::{HostMetadataAccount, HostMetadataFundingError};
use llguidance::toktrie::TokRxInfo;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct Account {
    refused: AtomicBool,
    charged: AtomicUsize,
}
#[derive(Debug)]
struct AccountHandle(Arc<Account>);
impl HostMetadataAccount for AccountHandle {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.0.refused.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        self.0.charged.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
struct Bytes(TokTrie);
impl TokenizerEnv for Bytes {
    fn tok_trie(&self) -> &TokTrie {
        &self.0
    }
    fn tokenize_bytes(&self, input: &[u8]) -> Vec<u32> {
        input.iter().map(|&byte| u32::from(byte)).collect()
    }
    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}
fn funding() -> (Arc<Account>, HostMetadataFunding) {
    let account = Arc::new(Account::default());
    let funding = HostMetadataFunding::new(AccountHandle(account.clone())).unwrap();
    (account, funding)
}
fn template() -> (Template, Arc<Account>, HostMetadataFunding) {
    template_with_memory(DependencyMemoryPolicy::default())
}
fn template_with_memory(
    memory: DependencyMemoryPolicy,
) -> (Template, Arc<Account>, HostMetadataFunding) {
    let (account, funding) = funding();
    let preparation = PreparationFunding::from_metadata(&funding).with_memory_policy(memory);
    let mut words = (0..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
    words.push(b"\xff<eos>".to_vec());
    let source = Arc::new(Bytes(TokTrie::from(&TokRxInfo::new(257, 256), &words)));
    let environment = Environment::new(TokenizerSource::Ordinary(source), &preparation).unwrap();
    let token_env: TokEnv = environment.clone();
    let mut factory = ParserFactory::new_simple(&token_env).unwrap();
    factory.quiet();
    let template = Template::compile(
        &factory,
        environment,
        TopLevelGrammar::from_lark("start: \"a\" (\"b\" | \"c\") \"d\"".into()),
        &preparation,
    )
    .unwrap();
    (template, account, funding)
}
fn mask(session: &mut Session) -> Vec<u32> {
    session.compute_mask().unwrap();
    (0..257)
        .filter(|&id| session.token_mask().unwrap().is_allowed(id))
        .collect()
}
#[test]
fn template_sessions_and_concurrent_snapshots_keep_independent_branches() {
    let (template, cold_account, cold_funding) = template();
    let (_, first_funding) = funding();
    let mut first = template.create_session(&first_funding).unwrap();
    assert_eq!(mask(&mut first), [97]);
    assert_eq!(first.try_consume_tokens(&[120]).unwrap(), 0);
    assert_eq!(first.try_consume_tokens(&[97]).unwrap(), 1);
    assert_eq!(mask(&mut first), [98, 99]);
    let (_, second_funding) = funding();
    let mut second = first.try_copy(&second_funding).unwrap();
    // A session must never tokenize through the compiler's diagnostic account.
    cold_account.refused.store(true, Ordering::SeqCst);
    drop(cold_funding);
    drop(template);
    assert!(!Arc::ptr_eq(&first.environment, &second.environment));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            assert_eq!(first.try_consume_tokens(&[98]).unwrap(), 1);
            assert_eq!(mask(&mut first), [100]);
            assert_eq!(first.try_consume_tokens(&[100]).unwrap(), 1);
            assert_eq!(first.parser.final_bytes(), b"abd");
            assert!(first.is_accepting().unwrap());
            assert_eq!(mask(&mut first), [256]);
        });
        scope.spawn(|| {
            assert_eq!(second.try_consume_tokens(&[99]).unwrap(), 1);
            assert_eq!(mask(&mut second), [100]);
            assert_eq!(second.try_consume_tokens(&[100]).unwrap(), 1);
            assert_eq!(second.parser.final_bytes(), b"acd");
            assert!(second.is_accepting().unwrap());
            assert_eq!(mask(&mut second), [256]);
        });
    });
    second.rollback(2).unwrap();
    assert_eq!(mask(&mut second), [98, 99]);
    assert_eq!(first.parser.final_bytes(), b"abd");
    second.reset().unwrap();
    assert_eq!(second.num_tokens(), 0);
    assert_eq!(mask(&mut second), [97]);
}
#[test]
fn refused_copy_and_operation_keep_original_state_and_destination_account() {
    let (template, _, cold_funding) = template();
    let (account, funding) = funding();
    let mut original = template.create_session(&funding).unwrap();
    original.try_consume_tokens(&[97]).unwrap();
    assert_eq!(mask(&mut original), [98, 99]);
    let charged = account.charged.load(Ordering::SeqCst);
    let expected = original.copy_required_bytes().unwrap();
    let copy = original.try_copy(&funding).unwrap();
    assert_eq!(account.charged.load(Ordering::SeqCst) - charged, expected);
    drop(copy);
    account.refused.store(true, Ordering::SeqCst);
    let copy_error = original.try_copy(&funding).unwrap_err();
    let operation_error = original.try_consume_tokens(&[98]).unwrap_err();
    assert_eq!(original.parser.final_bytes(), b"a");
    assert_eq!(original.num_tokens(), 1);
    let weak = Arc::downgrade(&account);
    drop((original, funding, account, template, cold_funding));
    assert!(weak.upgrade().is_some());
    drop(copy_error);
    assert!(weak.upgrade().is_some());
    drop(operation_error);
    assert!(weak.upgrade().is_none());
}

#[test]
fn sessions_and_snapshots_retain_selected_dependency_headroom() {
    let memory = DependencyMemoryPolicy {
        fixed_bytes: 2048,
        bytes_per_input_byte: 7,
    };
    let (template, _, _) = template_with_memory(memory);
    assert_eq!(template.memory_policy(), memory);
    let (_, account) = funding();
    let mut session = template.create_session(&account).unwrap();
    assert_eq!(session.funding.memory_policy(), memory);
    assert_eq!(session.environment.funding.memory_policy(), memory);
    session.try_consume_tokens(&[97]).unwrap();
    let expected = session.base_bytes
        + memory.estimate(1).unwrap()
        + eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<Environment>().unwrap();
    assert_eq!(session.copy_required_bytes(), Some(expected));
    let (_, destination) = funding();
    let mut copy = session.try_copy(&destination).unwrap();
    drop((session, template, account));
    assert_eq!(copy.funding.memory_policy(), memory);
    assert_eq!(copy.environment.funding.memory_policy(), memory);
    assert_eq!(mask(&mut copy), [98, 99]);
}
