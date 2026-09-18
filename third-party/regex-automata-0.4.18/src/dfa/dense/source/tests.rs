use super::*;
use crate::{nfa::thompson::pikevm::PikeVM, Anchored, Input};

#[test]
fn emitted_tables_match_independent_compiler_on_unicode_and_spans() {
    let mut texts = alloc::vec![
        alloc::string::String::new(),
        "The assistant's answer has 12345 tokens.\r\n".into(),
        "élève 東京🙂 १२३ e\u{301}\u{200c}\u{200d}\u{2028}".into(),
        "\r\n   \t x\r\n\n\\///!? 42".into(),
        " ".repeat(1024),
        "東京".repeat(512),
    ];
    let alphabet = [
        'a', 'Z', 'é', '東', '🙂', '\u{301}', '\u{200c}', '0', '९', ' ', '\r',
        '\n', '\t', '\'', '/', '!', '\0',
    ];
    let mut state = 0x92d68ca2u32;
    for _ in 0..256 {
        let mut text = alloc::string::String::new();
        for _ in 0..24 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            text.push(alphabet[state as usize % alphabet.len()]);
        }
        texts.push(text);
    }
    for recipe in RECIPES {
        let plan = Plan::new(recipe.pattern).unwrap();
        let heap = plan.heap_bytes();
        let dfa = plan.prepare().unwrap();
        assert_eq!(dfa.memory_usage() + mem::size_of::<DFA<Vec<u32>>>(), heap);
        let reference = PikeVM::new(recipe.pattern).unwrap();
        let mut cache = reference.create_cache();
        for text in &texts {
            for start in 0..=text.len().min(48) {
                let input = Input::new(text)
                    .span(start..text.len())
                    .anchored(Anchored::Yes);
                let mut slots = [None; 2];
                let expected = reference
                    .search_slots(&mut cache, &input, &mut slots)
                    .map(|pid| {
                        crate::HalfMatch::new(pid, slots[1].unwrap().get())
                    });
                assert_eq!(
                    dfa.try_search_fwd(&input).unwrap(),
                    expected,
                    "{:?}, {:?}, start={start}",
                    recipe.pattern,
                    text
                );
            }
        }
    }
}

#[test]
fn every_real_table_refusal_retains_prior_destinations() {
    for recipe in RECIPES {
        for (failed, buffer) in Buffer::ALL.into_iter().enumerate() {
            let plan = Plan::new(recipe.pattern).unwrap();
            let lengths = tables(&plan.borrowed).map(|t| t.len());
            let error = plan
                .fail_reserve_for_testing(buffer)
                .prepare()
                .expect_err("real reserve overflow");
            assert_eq!(error.buffer(), buffer);
            let capacities = error.capacities();
            for index in 0..5 {
                assert_eq!(
                    capacities[index],
                    if index < failed { lengths[index] } else { 0 }
                );
            }
            assert!(matches!(error.cause, Cause::Reserve(_)));
        }
    }
}
