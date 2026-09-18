//! Development-only fixed-input recipe emitter. No runtime admission API.
//!
//! The compiler hook is armed only by the ignored emitter test. Allocation
//! addresses correlate live compiler results and never enter emitted data.

use std::{cell::RefCell, fmt::Write as _, io::Write as _, path::Path};

use regex_automata::{
    nfa::thompson::{pikevm::PikeVM, State as NfaState},
    util::{look::Look, primitives::PatternID, syntax::Config as Syntax},
    MatchKind,
};

use super::{Body, Insn, Source};
use crate::{Assertion, RegexOptions};

// Exact selected sources from the retained tokenizer inventory, including
// existing GGUF preprocessing expressions. These are
// inputs to the ordinary compiler, not model-family dispatch or runtime tables.
const PATTERNS: &[&str] = &[
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"\w+|[^\w\s]+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\x{200C}|\x{200D})+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\u{00B2}\u{00B3}\u{00B9}\u{00BC}\u{00BD}\u{00BE}]+|[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\s]+",
];

struct Compiled {
    pattern: String,
    syntax: Syntax,
    states: *const NfaState,
    count: usize,
    inner: PikeVM,
}

std::thread_local! {
    static COMPILED: RefCell<Option<Vec<Compiled>>> = const { RefCell::new(None) };
}

pub(crate) fn record(pattern: &str, options: &crate::compile::CompileOptions, inner: &PikeVM) {
    COMPILED.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(records) = slot.as_mut() {
            assert_eq!(options.delegate_size_limit, None);
            assert_eq!(options.delegate_dfa_size_limit, None);
            records.push(Compiled {
                pattern: pattern.to_owned(),
                syntax: Syntax::new()
                    .unicode(options.unicode)
                    .utf8(options.bytes_mode == crate::BytesMode::Unicode),
                states: inner.get_nfa().states().as_ptr(),
                count: inner.get_nfa().states().len(),
                inner: inner.clone(),
            });
        }
    });
}

struct Recording;
impl Recording {
    fn start() -> Self {
        COMPILED.with(|slot| {
            let mut slot = slot.borrow_mut();
            assert!(slot.is_none(), "nested emitter recording");
            *slot = Some(Vec::new());
        });
        Self
    }
    fn finish(self) -> Vec<Compiled> {
        COMPILED.with(|slot| slot.borrow_mut().take().expect("active emitter"))
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        // Release the RefCell loan before destroying collected strings.
        let pending = COMPILED.with(|slot| slot.borrow_mut().take());
        drop(pending);
    }
}

fn syntax(config: Syntax) -> String {
    format!(
        "SyntaxRecipe {{ unicode: {}, case_insensitive: {}, multi_line: {}, \
         dot_matches_new_line: {}, crlf: {}, line_terminator: {}, swap_greed: {}, \
         ignore_whitespace: {}, utf8: {}, nest_limit: {}, octal: {} }}",
        config.get_unicode(),
        config.get_case_insensitive(),
        config.get_multi_line(),
        config.get_dot_matches_new_line(),
        config.get_crlf(),
        config.get_line_terminator(),
        config.get_swap_greed(),
        config.get_ignore_whitespace(),
        config.get_utf8(),
        config.get_nest_limit(),
        config.get_octal(),
    )
}

fn assert_defaults(options: &RegexOptions) {
    let defaults = RegexOptions::default();
    assert_eq!(syntax(options.syntaxc), syntax(defaults.syntaxc));
    assert_eq!(
        options.hard_regex_runtime_options.backtrack_limit,
        defaults.hard_regex_runtime_options.backtrack_limit
    );
    assert_eq!(options.delegate_size_limit, defaults.delegate_size_limit);
    assert_eq!(
        options.delegate_dfa_size_limit,
        defaults.delegate_dfa_size_limit
    );
    assert_eq!(options.oniguruma_mode, defaults.oniguruma_mode);
    assert_eq!(options.bytes_mode, defaults.bytes_mode);
    assert_eq!(
        options.ignore_numbered_groups_when_named_groups_exist,
        defaults.ignore_numbered_groups_when_named_groups_exist
    );
    assert!(options.seek_filter.is_none());
    assert_eq!(options.delegate_prefilter, defaults.delegate_prefilter);
    assert!(!options.hard_regex_runtime_options.find_not_empty);
    assert!(
        !options
            .hard_regex_runtime_options
            .disallow_empty_match_at_eof_after_newline
    );
    assert!(
        !options
            .hard_regex_runtime_options
            .allow_input_assertion_overrides
    );
}

fn look(look: Look) -> &'static str {
    match look {
        Look::Start => "Look::Start",
        Look::End => "Look::End",
        Look::StartLF => "Look::StartLF",
        Look::EndLF => "Look::EndLF",
        Look::StartCRLF => "Look::StartCRLF",
        Look::EndCRLF => "Look::EndCRLF",
        Look::WordAscii => "Look::WordAscii",
        Look::WordAsciiNegate => "Look::WordAsciiNegate",
        Look::WordUnicode => "Look::WordUnicode",
        Look::WordUnicodeNegate => "Look::WordUnicodeNegate",
        Look::WordStartAscii => "Look::WordStartAscii",
        Look::WordEndAscii => "Look::WordEndAscii",
        Look::WordStartUnicode => "Look::WordStartUnicode",
        Look::WordEndUnicode => "Look::WordEndUnicode",
        Look::WordStartHalfAscii => "Look::WordStartHalfAscii",
        Look::WordEndHalfAscii => "Look::WordEndHalfAscii",
        Look::WordStartHalfUnicode => "Look::WordStartHalfUnicode",
        Look::WordEndHalfUnicode => "Look::WordEndHalfUnicode",
    }
}

fn assertion(value: Assertion) -> String {
    match value {
        Assertion::StartText => "Assertion::StartText".to_owned(),
        Assertion::EndTextIgnoreTrailingNewlines { crlf } => format!(
            "Assertion::EndTextIgnoreTrailingNewlines {{ crlf: {} }}",
            crlf
        ),
        Assertion::StartLineOniguruma { crlf } => {
            format!("Assertion::StartLineOniguruma {{ crlf: {} }}", crlf)
        }
        Assertion::EndText => "Assertion::EndText".to_owned(),
        Assertion::StartLine { crlf } => format!("Assertion::StartLine {{ crlf: {} }}", crlf),
        Assertion::EndLine { crlf } => format!("Assertion::EndLine {{ crlf: {} }}", crlf),
        Assertion::LeftWordBoundary => "Assertion::LeftWordBoundary".to_owned(),
        Assertion::LeftWordHalfBoundary => "Assertion::LeftWordHalfBoundary".to_owned(),
        Assertion::RightWordBoundary => "Assertion::RightWordBoundary".to_owned(),
        Assertion::RightWordHalfBoundary => "Assertion::RightWordHalfBoundary".to_owned(),
        Assertion::WordBoundary => "Assertion::WordBoundary".to_owned(),
        Assertion::NotWordBoundary => "Assertion::NotWordBoundary".to_owned(),
    }
}

#[derive(Default)]
struct Counts {
    states: usize,
    sparse_states: usize,
    transitions: usize,
    union_states: usize,
    union_edges: usize,
    epsilon_edges: usize,
    groups: usize,
    slots: usize,
}

fn nfa(record: &Compiled, inner: &PikeVM) -> (String, Counts) {
    let nfa = inner.get_nfa();
    assert_eq!(record.states, nfa.states().as_ptr());
    assert_eq!(record.count, nfa.states().len());
    assert_eq!(
        inner.get_config().get_match_kind(),
        MatchKind::LeftmostFirst
    );
    assert!(inner.get_config().get_prefilter().is_none());
    assert!(!nfa.is_reverse());
    assert_eq!(nfa.pattern_len(), 1, "initial workspace profile");
    let mut out = String::new();
    let mut counts = Counts {
        states: nfa.states().len(),
        slots: nfa.group_info().slot_len(),
        ..Counts::default()
    };
    // Debug is used only for Rust string literals, never NFA/program dumps.
    writeln!(
        out,
        "NfaRecipe {{ pattern: {:?}, syntax: {},",
        record.pattern,
        syntax(record.syntax)
    )
    .unwrap();
    writeln!(
        out,
        "anchored: {}, unanchored: {}, utf8: {}, reverse: {}, line_terminator: {},",
        nfa.start_anchored().as_usize(),
        nfa.start_unanchored().as_usize(),
        nfa.is_utf8(),
        nfa.is_reverse(),
        nfa.look_matcher().get_line_terminator()
    )
    .unwrap();
    out.push_str("starts: &[");
    for index in 0..nfa.pattern_len() {
        write!(
            out,
            "{},",
            nfa.start_pattern(PatternID::must(index))
                .unwrap()
                .as_usize()
        )
        .unwrap();
    }
    out.push_str("], groups: &[\n");
    for index in 0..nfa.pattern_len() {
        let pid = PatternID::must(index);
        out.push_str("&[");
        for group in 0..nfa.group_info().group_len(pid) {
            assert!(
                nfa.group_info().to_name(pid, group).is_none(),
                "unexpected named capture"
            );
            let (start, end) = nfa.group_info().slots(pid, group).unwrap();
            write!(out, "({},{}),", start, end).unwrap();
            counts.groups += 1;
        }
        out.push_str("],\n");
    }
    out.push_str("], states: &[\n");
    for state in nfa.states() {
        match state {
            NfaState::ByteRange { trans } => writeln!(
                out,
                "StateRecipe::ByteRange({}, {}, {}),",
                trans.start,
                trans.end,
                trans.next.as_usize()
            )
            .unwrap(),
            NfaState::Sparse(sparse) => {
                counts.sparse_states += 1;
                counts.transitions += sparse.transitions.len();
                out.push_str("StateRecipe::Sparse(&[");
                for trans in sparse.transitions.iter() {
                    write!(
                        out,
                        "({}, {}, {}),",
                        trans.start,
                        trans.end,
                        trans.next.as_usize()
                    )
                    .unwrap();
                }
                out.push_str("]),\n");
            }
            NfaState::Dense(_) => {
                panic!("unexpected dense state: constructor profile must be reviewed")
            }
            NfaState::Look { look: value, next } => {
                counts.epsilon_edges += 1;
                writeln!(
                    out,
                    "StateRecipe::Look({}, {}),",
                    look(*value),
                    next.as_usize()
                )
                .unwrap();
            }
            NfaState::Union { alternates } => {
                counts.union_states += 1;
                counts.union_edges += alternates.len();
                counts.epsilon_edges += alternates.len();
                out.push_str("StateRecipe::Union(&[");
                for next in alternates.iter() {
                    write!(out, "{},", next.as_usize()).unwrap();
                }
                out.push_str("]),\n");
            }
            NfaState::BinaryUnion { alt1, alt2 } => {
                counts.epsilon_edges += 2;
                writeln!(
                    out,
                    "StateRecipe::BinaryUnion({}, {}),",
                    alt1.as_usize(),
                    alt2.as_usize()
                )
                .unwrap();
            }
            NfaState::Capture {
                next,
                pattern_id,
                group_index,
                slot,
            } => {
                counts.epsilon_edges += 1;
                writeln!(
                    out,
                    "StateRecipe::Capture {{ next: {}, pattern: {}, group: {}, slot: {} }},",
                    next.as_usize(),
                    pattern_id.as_usize(),
                    group_index.as_usize(),
                    slot.as_usize()
                )
                .unwrap();
            }
            NfaState::Fail => out.push_str("StateRecipe::Fail,\n"),
            NfaState::Match { pattern_id } => {
                writeln!(out, "StateRecipe::Match({}),", pattern_id.as_usize()).unwrap()
            }
        }
    }
    writeln!(
        out,
        "], expected: PropertiesRecipe {{ has_empty: {}, has_capture: {}, look_any: &[",
        nfa.has_empty(),
        nfa.has_capture()
    )
    .unwrap();
    for value in nfa.look_set_any().iter() {
        write!(out, "{},", look(value)).unwrap();
    }
    out.push_str("], look_prefix: &[");
    for value in nfa.look_set_prefix_any().iter() {
        write!(out, "{},", look(value)).unwrap();
    }
    out.push_str("], byte_classes: &[");
    for byte in 0..=255u8 {
        write!(out, "{},", nfa.byte_classes().get(byte)).unwrap();
    }
    out.push_str("] } },\n");
    (out, counts)
}

fn bound(value: usize) -> String {
    if value == usize::MAX {
        "usize::MAX".to_owned()
    } else {
        value.to_string()
    }
}

fn instruction(out: &mut String, insn: &Insn, records: &[Compiled], ordinal: &mut usize) {
    match insn {
        Insn::End => out.push_str("InsnRecipe::End"),
        Insn::Any => out.push_str("InsnRecipe::Any"),
        Insn::AnyNoNL => out.push_str("InsnRecipe::AnyNoNL"),
        Insn::AnyNoCRLF => out.push_str("InsnRecipe::AnyNoCRLF"),
        Insn::CharClass(crate::vm::CharClassMatcher::Codepoint(ranges)) => {
            write!(out, "InsnRecipe::CharClassCodepoint(&{:?})", ranges).unwrap()
        }
        Insn::CharClass(crate::vm::CharClassMatcher::Byte(ranges)) => {
            write!(out, "InsnRecipe::CharClassByte(&{:?})", ranges).unwrap()
        }
        Insn::Assertion(value) => {
            write!(out, "InsnRecipe::Assertion({})", assertion(*value)).unwrap()
        }
        Insn::Lit(value) => write!(out, "InsnRecipe::Lit({:?})", value).unwrap(),
        Insn::SplitUnanchored(x, y) => {
            write!(out, "InsnRecipe::SplitUnanchored({}, {})", x, y).unwrap()
        }
        Insn::SaveCaptureGroupStart(x) => {
            write!(out, "InsnRecipe::SaveCaptureGroupStart({})", x).unwrap()
        }
        Insn::Split(x, y) => write!(out, "InsnRecipe::Split({}, {})", x, y).unwrap(),
        Insn::Jmp(x) => write!(out, "InsnRecipe::Jmp({})", x).unwrap(),
        Insn::Save(x) => write!(out, "InsnRecipe::Save({})", x).unwrap(),
        Insn::Save0(x) => write!(out, "InsnRecipe::Save0({})", x).unwrap(),
        Insn::Restore(x) => write!(out, "InsnRecipe::Restore({})", x).unwrap(),
        Insn::RepeatGr {
            lo,
            hi,
            next,
            repeat,
        } => write!(
            out,
            "InsnRecipe::RepeatGr {{ lo: {}, hi: {}, next: {}, repeat: {} }}",
            lo,
            bound(*hi),
            next,
            repeat
        )
        .unwrap(),
        Insn::RepeatNg {
            lo,
            hi,
            next,
            repeat,
        } => write!(
            out,
            "InsnRecipe::RepeatNg {{ lo: {}, hi: {}, next: {}, repeat: {} }}",
            lo,
            bound(*hi),
            next,
            repeat
        )
        .unwrap(),
        Insn::RepeatEpsilonGr {
            lo,
            next,
            repeat,
            check,
        } => write!(
            out,
            "InsnRecipe::RepeatEpsilonGr {{ lo: {}, next: {}, repeat: {}, check: {} }}",
            lo, next, repeat, check
        )
        .unwrap(),
        Insn::RepeatEpsilonNg {
            lo,
            next,
            repeat,
            check,
        } => write!(
            out,
            "InsnRecipe::RepeatEpsilonNg {{ lo: {}, next: {}, repeat: {}, check: {} }}",
            lo, next, repeat, check
        )
        .unwrap(),
        Insn::FailNegativeLookAround => out.push_str("InsnRecipe::FailNegativeLookAround"),
        Insn::DfaDelegate(_) => {
            write!(
                out,
                "InsnRecipe::Regular {{ pattern: {:?}, ordinal: {} }}",
                records[*ordinal].pattern, *ordinal
            )
            .unwrap();
            *ordinal += 1;
        }
        Insn::PikeDelegate(_) => {
            panic!("fixed construction inventory requires capture-free regular delegates")
        }
        _ => panic!("unexpected instruction: constructor profile must be reviewed"),
    }
    out.push_str(",\n");
}

fn write_new(directory: &Path, name: &str, text: &str) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .expect("new emitter output");
    file.write_all(text.as_bytes())
        .expect("complete emitter output");
}

#[test]
#[ignore = "development emitter; requires a fresh EREDU_REGEX_RECIPE_OUTPUT directory"]
fn emit_fixed_source_recipes() {
    let destination = std::env::var_os("EREDU_REGEX_RECIPE_OUTPUT").expect("explicit output path");
    let destination = Path::new(&destination);
    assert!(
        !destination.exists(),
        "never overwrite prior emitted evidence"
    );
    let mut sources = String::from("// Generated by the pinned ordinary compiler; no runtime addresses.\nstatic SOURCES: &[SourceRecipe] = &[\n");
    let mut unique: Vec<(String, String)> = Vec::new();
    let mut dfas = Vec::new();
    let mut inventory = String::from("{\n\"schema\": 1, \"kind\": \"compiler output counts, not allocation requirements\",\n\"sources\": [\n");
    for (source_index, pattern) in PATTERNS.iter().enumerate() {
        let recording = Recording::start();
        // Emit the existing explicit-workspace compiler for every selected
        // source, including regular expressions whose public wrapper can choose
        // a direct delegate. No instruction is synthesized by this emitter.
        let options = RegexOptions::default();
        let parsed = crate::source::ParsedSource::new(pattern, &options).unwrap();
        let info = parsed.analyze().unwrap();
        let program = crate::compile::compile_workspace(&info, parsed.compile_options(&options)).unwrap();
        let source = Source { body: Body::Fancy(program), options, pattern: (*pattern).to_owned() };
        let records = recording.finish();
        assert_defaults(&source.options);
        assert_eq!(source.pattern, *pattern);
        assert_eq!(
            match &source.body {
                Body::Wrap { .. } => 1,
                Body::Fancy(prog) => prog
                    .body
                    .iter()
                    .filter(|i| matches!(i, Insn::PikeDelegate(_) | Insn::DfaDelegate(_)))
                    .count(),
            },
            records.len()
        );
        if source_index != 0 {
            inventory.push_str(",\n");
        }
        write!(inventory, "{{\"source\":{},\"delegates\":[", source_index).unwrap();
        for (index, record) in records.iter().enumerate() {
            let inner = &record.inner;
            let (text, counts) = nfa(record, inner);
            if let Some((_, prior)) = unique
                .iter()
                .find(|(pattern, _)| pattern == &record.pattern)
            {
                assert_eq!(
                    prior, &text,
                    "same cooked/default source must emit identical NFA data"
                );
            } else {
                unique.push((record.pattern.clone(), text));
                let dfa = regex_automata::dfa::dense::DFA::builder()
                    .configure(
                        regex_automata::dfa::dense::DFA::config()
                            .start_kind(regex_automata::dfa::StartKind::Anchored)
                            .minimize(true),
                    )
                    .build_from_nfa(inner.get_nfa())
                    .expect("capture-free DFA");
                dfas.push((record.pattern.clone(), dfa));
            }
            if index != 0 {
                inventory.push(',');
            }
            write!(inventory, "{{\"ordinal\":{},\"states\":{},\"sparse_states\":{},\"transitions\":{},\"union_states\":{},\"union_edges\":{},\"epsilon_edges\":{},\"groups\":{},\"slots\":{}}}",
                index, counts.states, counts.sparse_states, counts.transitions, counts.union_states,
                counts.union_edges, counts.epsilon_edges, counts.groups, counts.slots).unwrap();
        }
        writeln!(
            sources,
            "SourceRecipe {{ pattern: {:?}, syntax: {}, backtrack_limit: {},",
            pattern,
            syntax(source.options.syntaxc),
            source.options.hard_regex_runtime_options.backtrack_limit
        )
        .unwrap();
        match &source.body {
            Body::Wrap { .. } => panic!("fixed inventory must retain fancy instructions"),
            Body::Fancy(prog) => {
                writeln!(sources, "saves: {}, instructions: &[", prog.n_saves).unwrap();
                let mut ordinal = 0;
                for insn in &prog.body {
                    instruction(&mut sources, insn, &records, &mut ordinal);
                }
                assert_eq!(ordinal, records.len());
                sources.push_str("] },\n");
                let literal_bytes: usize = prog
                    .body
                    .iter()
                    .filter_map(|insn| match insn {
                        Insn::Lit(s) => Some(s.len()),
                        _ => None,
                    })
                    .sum();
                write!(
                    inventory,
                    "],\"instructions\":{},\"saves\":{},\"literal_bytes\":{}}}",
                    prog.body.len(),
                    prog.n_saves,
                    literal_bytes
                )
                .unwrap();
            }
        }
    }
    sources.push_str("];\n");
    let mut nfas = String::from("// Generated final states; named/dense profiles are deliberately rejected.\nstatic NFAS: &[NfaRecipe] = &[\n");
    for (_, text) in &unique {
        nfas.push_str(text);
    }
    nfas.push_str("];\n");
    write!(
        inventory,
        "\n],\"unique_default_delegate_sources\":{}\n}}\n",
        unique.len()
    )
    .unwrap();
    std::fs::create_dir(destination).expect("fresh emitter directory");
    write_new(destination, "fancy-recipes.rs", &sources);
    write_new(destination, "pikevm-recipes.rs", &nfas);
    write_new(destination, "inventory.json", &inventory);
    let mut recipes =
        String::from("// Generated by emit_fixed_source_recipes from the pinned compiler.\n");
    for (index, (_, dfa)) in dfas.iter().enumerate() {
        for (endian, (data, padding)) in [
            ("little", dfa.to_bytes_little_endian()),
            ("big", dfa.to_bytes_big_endian()),
        ] {
            let data = &data[padding..];
            std::fs::write(destination.join(format!("{index}.{endian}.dfa")), data).unwrap();
            writeln!(recipes, "#[cfg(target_endian = {:?})]\nstatic D{index}: Aligned<{}> = Aligned(*include_bytes!({:?}));", endian, data.len(), format!("{index}.{endian}.dfa")).unwrap();
        }
    }
    recipes.push_str("static RECIPES: &[Recipe] = &[\n");
    for (index, (pattern, _)) in dfas.iter().enumerate() {
        writeln!(
            recipes,
            "Recipe {{ pattern: {:?}, bytes: &D{index}.0 }},",
            pattern
        )
        .unwrap();
    }
    recipes.push_str("];\n");
    write_new(destination, "dfa-recipes.rs", &recipes);
}
