use super::*;
use derivre::RegexBuilder;

fn regexes(patterns: &[&str]) -> RegexVec {
    regexes_with_limits(patterns, &mut ParserLimits::default()).unwrap()
}

fn regexes_with_limits(patterns: &[&str], limits: &mut ParserLimits) -> Result<RegexVec> {
    let mut builder = RegexBuilder::new();
    let regexes = patterns
        .iter()
        .map(|pattern| RxLexeme {
            rx: builder.mk_regex(pattern).unwrap(),
            lazy: false,
            priority: 0,
        })
        .collect();
    RegexVec::new_with_exprset(builder.exprset().clone(), regexes, None, limits)
}

fn selected(regexes: &RegexVec, indices: &[usize]) -> LexemeSet {
    let mut set = LexemeSet::new(regexes.rx_list.len());
    for &index in indices {
        set.add(LexemeIdx::new(index));
    }
    set
}

#[derive(Debug, PartialEq, Eq)]
struct Tables {
    rows: Vec<StateID>,
    row_capacity: usize,
    descriptors: usize,
    descriptor_capacity: usize,
    sets: Vec<Vec<u32>>,
    set_capacity: usize,
}
fn tables(regexes: &RegexVec) -> Tables {
    Tables {
        rows: regexes.state_table.clone(),
        row_capacity: regexes.state_table.capacity(),
        descriptors: regexes.state_descs.len(),
        descriptor_capacity: regexes.state_descs.capacity(),
        sets: (0..regexes.rx_sets.len())
            .map(|id| regexes.rx_sets.get(id as u32).to_vec())
            .collect(),
        set_capacity: regexes.rx_sets.retained_capacity_bytes().unwrap(),
    }
}

#[test]
fn impossible_construction_limits_precede_relevance_work_and_leave_fuel_unchanged() {
    for configured in [0, 1] {
        let mut limits = ParserLimits {
            max_lexer_states: configured,
            initial_lexer_fuel: 0,
            ..ParserLimits::default()
        };
        let error = regexes_with_limits(&["[a-z]{3,8}", "[0-9]+"], &mut limits).unwrap_err();
        let cause = error
            .downcast_ref::<crate::api::InvalidLexerStateLimit>()
            .unwrap();
        assert_eq!(cause.configured, configured);
        assert_eq!(cause.minimum, 2);
        assert_eq!(limits.initial_lexer_fuel, 0);
    }
}

#[test]
fn construction_limit_covers_reserved_states_and_rejects_the_first_new_state() {
    let mut limits = ParserLimits {
        max_lexer_states: 2,
        ..ParserLimits::default()
    };
    let mut regexes = regexes_with_limits(&["abc"], &mut limits).unwrap();
    assert_eq!(regexes.max_states, 2);
    assert_eq!(regexes.state_descs.len(), 2);
    assert_eq!(regexes.rx_sets.len(), 2);
    assert_eq!(regexes.state_table.len(), 2 * regexes.alpha.len());
    assert!(!regexes.has_error());
    let before = tables(&regexes);
    assert_eq!(
        regexes.initial_state(&selected(&regexes, &[0])),
        StateID::DEAD
    );
    assert_eq!(tables(&regexes), before);
    assert_eq!(
        regexes.get_error().as_deref(),
        Some("too many states: 2 >= 2")
    );
}

#[test]
fn finite_construction_limit_and_error_are_preserved_by_independent_clones() {
    let mut limits = ParserLimits {
        max_lexer_states: 3,
        ..ParserLimits::default()
    };
    let mut original = regexes_with_limits(&["abc", "def"], &mut limits).unwrap();
    let first = selected(&original, &[0]);
    let other = selected(&original, &[1]);
    let state = original.initial_state(&first);
    assert!(!state.is_dead());
    let mut copied = original.clone();
    assert_eq!(copied.max_states, 3);
    assert_eq!(copied.initial_state(&first), state);
    let before = tables(&copied);
    assert_eq!(copied.initial_state(&other), StateID::DEAD);
    assert_eq!(tables(&copied), before);
    assert!(!original.has_error());
    assert_eq!(original.initial_state(&first), state);
    let mut failed_copy = copied.clone();
    assert_eq!(failed_copy.get_error(), copied.get_error());
    failed_copy.set_max_states(32);
    assert_eq!(failed_copy.initial_state(&other), StateID::DEAD);
    assert_eq!(tables(&failed_copy), before);
}

#[test]
fn unseen_initial_state_is_rejected_before_hashcons_descriptor_or_row_growth() {
    let mut regexes = regexes(&["abc", "def", "ghi"]);
    let selection = selected(&regexes, &[0, 1, 2]);
    regexes.set_max_states(2); // The two preexisting sentinel states.
    let before = tables(&regexes);
    assert_eq!(regexes.initial_state(&selection), StateID::DEAD);
    assert!(regexes.has_error());
    assert_eq!(
        regexes.get_error().as_deref(),
        Some("too many states: 2 >= 2")
    );
    assert_eq!(tables(&regexes), before);
    assert_eq!(regexes.initial_state(&selection), StateID::DEAD);
    assert_eq!(tables(&regexes), before);
    assert!(regexes.state_desc(StateID::DEAD).possible.is_empty());
}

#[test]
fn exact_limit_and_lowered_limits_preserve_existing_ids_until_a_new_state_is_needed() {
    let mut regexes = regexes(&["abc", "def"]);
    let first = selected(&regexes, &[0]);
    let other = selected(&regexes, &[1]);
    regexes.set_max_states(3);
    let state = regexes.initial_state(&first);
    assert_eq!(state.as_usize(), 2);
    assert_eq!(regexes.state_descs.len(), 3);
    assert!(!regexes.has_error());
    let before = tables(&regexes);
    for limit in [3, 2, 1, 0] {
        regexes.set_max_states(limit);
        assert_eq!(regexes.initial_state(&first), state);
        assert_eq!(regexes.limit_state_to(state, &first), state);
        assert_eq!(regexes.initial_state(&LexemeSet::new(2)), StateID::DEAD);
        assert_eq!(tables(&regexes), before);
        assert!(!regexes.has_error());
    }
    assert_eq!(regexes.initial_state(&other), StateID::DEAD);
    assert_eq!(tables(&regexes), before);
    assert_eq!(
        regexes.get_error().as_deref(),
        Some("too many states: 3 >= 0")
    );
}

#[test]
fn uncached_transition_to_an_existing_state_succeeds_at_the_limit() {
    let mut regexes = regexes(&["a*"]);
    let state = regexes.initial_state(&selected(&regexes, &[0]));
    regexes.set_max_states(regexes.state_descs.len());
    let count = regexes.state_descs.len();
    let sets = regexes.rx_sets.len();
    let slot = regexes.alpha.map_state(state, b'a');
    assert_eq!(regexes.state_table[slot], StateID::MISSING);
    let next = regexes.transition(state, b'a');
    assert_eq!(next, state);
    assert_eq!(regexes.transition(state, b'a'), next);
    assert!(!regexes.has_error());
    assert_eq!(regexes.state_descs.len(), count);
    assert_eq!(regexes.rx_sets.len(), sets);
    assert_eq!(regexes.num_transitions, 1);
}

#[test]
fn transition_miss_at_limit_returns_dead_without_publishing_a_dangling_state() {
    let mut regexes = regexes(&["abc"]);
    let state = regexes.initial_state(&selected(&regexes, &[0]));
    regexes.set_max_states(regexes.state_descs.len());
    let before = tables(&regexes);
    let next = regexes.transition(state, b'a');
    assert_eq!(next, StateID::DEAD);
    assert!(regexes.has_error());
    let after = tables(&regexes);
    assert_eq!(after.descriptors, before.descriptors);
    assert_eq!(after.descriptor_capacity, before.descriptor_capacity);
    assert_eq!(after.row_capacity, before.row_capacity);
    assert_eq!(after.rows.len(), before.rows.len());
    assert_eq!(after.sets, before.sets);
    assert_eq!(after.set_capacity, before.set_capacity);
    assert!(after
        .rows
        .iter()
        .all(|state| state.as_usize() < after.descriptors));
    assert_eq!(regexes.state_desc(next).state, StateID::DEAD);
    assert_eq!(regexes.transition(state, b'b'), StateID::DEAD);
    assert_eq!(tables(&regexes), after);
}

#[test]
fn selecting_an_unseen_subset_uses_the_same_limit_guard() {
    let mut regexes = regexes(&["abc", "def"]);
    let state = regexes.initial_state(&selected(&regexes, &[0, 1]));
    regexes.set_max_states(regexes.state_descs.len());
    let first = selected(&regexes, &[0]);
    let before = tables(&regexes);
    assert_eq!(regexes.limit_state_to(state, &first), StateID::DEAD);
    assert!(regexes.has_error());
    assert_eq!(tables(&regexes), before);
}

#[test]
fn fuel_failure_keeps_its_error_and_does_not_publish_partial_derivative_state() {
    let mut regexes = regexes(&["abc", "ade"]);
    let state = regexes.initial_state(&selected(&regexes, &[0, 1]));
    let before = tables(&regexes);
    regexes.set_fuel(0);
    assert_eq!(regexes.transition(state, b'a'), StateID::DEAD);
    assert!(regexes.has_error());
    assert_eq!(
        regexes.get_error().as_deref(),
        Some("too many expressions constructed")
    );
    let after = tables(&regexes);
    assert_eq!(after.sets, before.sets);
    assert_eq!(after.set_capacity, before.set_capacity);
    assert_eq!(after.descriptors, before.descriptors);
    assert_eq!(after.descriptor_capacity, before.descriptor_capacity);
    assert_eq!(after.rows.len(), before.rows.len());
    regexes.set_fuel(u64::MAX);
    regexes.set_max_states(usize::MAX);
    assert_eq!(regexes.get_fuel(), 0);
    assert_eq!(regexes.transition(state, b'a'), StateID::DEAD);
    assert_eq!(tables(&regexes), after);
}

#[test]
fn state_and_transition_and_descriptor_geometry_are_checked_without_allocating() {
    assert_eq!(checked_state_table_growth(2, 256, 3), Ok(768));
    assert_eq!(
        checked_state_table_growth(3, 256, 3),
        Err(StateGrowthError::Limit {
            states: 3,
            limit: 3
        })
    );
    assert_eq!(
        checked_state_table_growth(2, 0, 3),
        Err(StateGrowthError::TableCapacity)
    );
    assert_eq!(
        checked_state_table_growth(2, usize::MAX, 3),
        Err(StateGrowthError::TableCapacity)
    );
    let invalid_id = (u32::MAX as usize >> 1) + 1;
    assert_eq!(
        checked_state_table_growth(invalid_id, 1, usize::MAX),
        Err(StateGrowthError::StateId)
    );
    // On 32-bit hosts descriptor capacity can fail before a one-symbol row.
    let descriptor_overflow = isize::MAX as usize / std::mem::size_of::<StateDesc>();
    if descriptor_overflow <= (u32::MAX >> 1) as usize {
        assert_eq!(
            checked_state_table_growth(descriptor_overflow, 1, usize::MAX),
            Err(StateGrowthError::TableCapacity)
        );
    }
    let row_overflow = isize::MAX as usize / std::mem::size_of::<StateID>();
    assert_eq!(
        checked_state_table_growth(1, row_overflow, usize::MAX),
        Err(StateGrowthError::TableCapacity)
    );
}

#[test]
fn adequate_limit_preserves_nonzero_matching_across_deep_cloned_regex_state() {
    let mut original = regexes(&["abc", "abd"]);
    original.set_max_states(32);
    let start = original.initial_state(&selected(&original, &[0, 1]));
    let mut copied = original.clone();
    for (regexes, word, expected) in [(&mut original, b"abc", 0), (&mut copied, b"abd", 1)] {
        let mut state = start;
        for &byte in word {
            state = regexes.transition(state, byte);
        }
        assert!(!state.is_dead());
        assert!(!regexes.has_error());
        assert!(regexes
            .state_desc(state)
            .greedy_accepting
            .contains(LexemeIdx::new(expected)));
        assert!(!regexes
            .state_desc(state)
            .greedy_accepting
            .contains(LexemeIdx::new(1 - expected)));
        assert_eq!(
            regexes.state_table.len(),
            regexes.state_descs.len() * regexes.alpha.len()
        );
        assert_eq!(regexes.rx_sets.len(), regexes.state_descs.len());
    }
}
