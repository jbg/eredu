//! Counts completed ParserState copies, not allocator bytes or a parser budget.

use super::*;
use crate::{
    api::{GrammarInit, LLGuidanceOptions, NodeProps},
    grammar_builder::GrammarBuilder,
    ParserFactory, TokenParser,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use toktrie::ApproximateTokEnv;

#[derive(Default)]
struct CopyCounts {
    clones: AtomicUsize,
    live: AtomicUsize,
    peak: AtomicUsize,
    drops: AtomicUsize,
}

impl CopyCounts {
    fn snapshot(&self) -> (usize, usize, usize, usize) {
        (
            self.clones.load(Ordering::SeqCst),
            self.live.load(Ordering::SeqCst),
            self.peak.load(Ordering::SeqCst),
            self.drops.load(Ordering::SeqCst),
        )
    }
}

// Detached for every unrelated fixture. Attaching after real parsing ensures
// the witness counts copies of nonempty state, including captures and history.
#[derive(Default)]
pub(super) struct CloneWitness(Option<Arc<CopyCounts>>);

impl CloneWitness {
    fn attach() -> (Self, Arc<CopyCounts>) {
        let counts = Arc::new(CopyCounts::default());
        counts.live.store(1, Ordering::SeqCst);
        counts.peak.store(1, Ordering::SeqCst);
        (Self(Some(Arc::clone(&counts))), counts)
    }
}

impl Clone for CloneWitness {
    fn clone(&self) -> Self {
        if let Some(counts) = &self.0 {
            counts.clones.fetch_add(1, Ordering::SeqCst);
            let live = counts.live.fetch_add(1, Ordering::SeqCst) + 1;
            counts.peak.fetch_max(live, Ordering::SeqCst);
        }
        Self(self.0.clone())
    }
}

impl Drop for CloneWitness {
    fn drop(&mut self) {
        if let Some(counts) = &self.0 {
            counts.live.fetch_sub(1, Ordering::SeqCst);
            counts.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

fn consume(parser: &mut TokenParser, bytes: &[u8]) {
    for &byte in bytes {
        let mask = parser.compute_mask().unwrap();
        assert!(mask.is_allowed(u32::from(byte)), "rejected byte {byte:?}");
        assert_eq!(parser.consume_token(u32::from(byte)).unwrap(), 0);
        parser.check_stop().unwrap();
    }
}

fn populated_parser() -> TokenParser {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
    factory.quiet();
    factory.limits_mut().precompute_large_lexemes = false;
    let mut builder = GrammarBuilder::new(None, factory.limits().clone());
    builder
        .add_grammar(LLGuidanceOptions::default(), RegexAst::NoMatch)
        .unwrap();
    let name_rx = builder.regex.regex("[a-z]{3,8}").unwrap();
    let name = builder.lexeme_ext(
        name_rx,
        None,
        NodeProps {
            capture_name: Some("name".to_owned()),
            ..NodeProps::default()
        },
    );
    let separator = builder.string(":");
    let value_rx = builder.regex.regex("[0-9]{2}").unwrap();
    let value = builder.lexeme_ext(
        value_rx,
        None,
        NodeProps {
            capture_name: Some("value".to_owned()),
            ..NodeProps::default()
        },
    );
    let end = builder.string("\n");
    let sequence = builder.join(&[name, separator, value, end]);
    builder.set_start_node(sequence);
    let mut parser = factory
        .create_parser_from_init_default(GrammarInit::Internal(builder.grammar, builder.regex.spec))
        .unwrap();
    parser.start_without_prompt();
    consume(&mut parser, b"market:");
    assert_eq!(parser.get_capture("name"), Some(b"market".as_slice()));
    assert_eq!(parser.parser.get_bytes(), b"market:");
    assert!(!parser.parser.state.rows.is_empty());
    assert!(!parser.parser.state.row_infos.is_empty());
    assert!(!parser.parser.state.scratch.items.is_empty());
    assert!(!parser.parser.state.lexer_stack.is_empty());
    assert!(!parser.parser.state.captures.capture_list.is_empty());
    parser
}

fn assert_one_copy_and_independent_payload(deep: bool) {
    let mut source = populated_parser();
    let (witness, counts) = CloneWitness::attach();
    source.parser.state.copy_witness = witness;
    let mut copy = if deep {
        source.deep_clone()
    } else {
        source.clone()
    };

    // The old TokenParser deep-copy path produces two copies here: its first
    // temporary ordinary ParserState remains alive while the second is made.
    assert_eq!(counts.snapshot(), (1, 2, 2, 0));
    assert_eq!(
        Arc::ptr_eq(&source.parser.shared, &copy.parser.shared),
        !deep
    );
    assert!(Arc::ptr_eq(
        &source.parser.state.grammar,
        &copy.parser.state.grammar
    ));
    assert_eq!(source.parser.get_bytes(), copy.parser.get_bytes());
    assert_ne!(
        source.parser.state.bytes.as_ptr(),
        copy.parser.state.bytes.as_ptr()
    );
    assert_ne!(
        source.parser.state.rows.as_ptr(),
        copy.parser.state.rows.as_ptr()
    );
    assert_ne!(
        source.get_capture("name").unwrap().as_ptr(),
        copy.get_capture("name").unwrap().as_ptr(),
    );
    assert_eq!(source.captures(), copy.captures());
    let source_mask = source.compute_mask().unwrap();
    let copy_mask = copy.compute_mask().unwrap();
    assert_eq!(
        source_mask.iter().collect::<Vec<_>>(),
        copy_mask.iter().collect::<Vec<_>>()
    );
    assert!(copy_mask.is_allowed(u32::from(b'1')));
    assert!(!copy_mask.is_allowed(u32::from(b'x')));

    consume(&mut copy, b"17\n");
    assert!(copy.is_accepting());
    assert_eq!(copy.get_capture("value"), Some(b"17".as_slice()));
    assert_eq!(source.parser.get_bytes(), b"market:");
    assert_eq!(source.get_capture("value"), None);
    assert_eq!(source.get_capture("name"), Some(b"market".as_slice()));
    consume(&mut source, b"23\n");
    assert!(source.is_accepting());
    assert_eq!(source.get_capture("value"), Some(b"23".as_slice()));
    assert_eq!(copy.get_capture("value"), Some(b"17".as_slice()));
    assert_eq!(copy.parser.get_bytes(), b"market:17\n");
    assert_eq!(source.parser.get_bytes(), b"market:23\n");
    assert_eq!(counts.snapshot(), (1, 2, 2, 0));

    drop(source);
    assert_eq!(counts.snapshot(), (1, 1, 2, 1));
    assert_eq!(copy.get_capture("name"), Some(b"market".as_slice()));
    assert_eq!(copy.get_capture("value"), Some(b"17".as_slice()));
    drop(copy);
    assert_eq!(counts.snapshot(), (1, 0, 2, 2));
}

#[test]
fn populated_token_parser_deep_clone_has_one_state_copy_and_two_live_payloads() {
    assert_one_copy_and_independent_payload(true);
}

#[test]
fn populated_token_parser_ordinary_clone_copies_state_once_and_shares_the_lexer() {
    assert_one_copy_and_independent_payload(false);
}
