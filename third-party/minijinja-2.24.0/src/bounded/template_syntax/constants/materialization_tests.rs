//! Behavioral gates for the actual shared fold and retained metadata/payload destinations.
#![forbid(unsafe_code)]
use super::*;
use crate::value::primitive::text::{self as kernels, RepeatFailure};
use std::fmt::Write;

#[test]
fn materialized_text_operations_and_segmented_literals_match_prechange_oracle() {
    for source in [
        "''",
        "'a' 'λ' '\\n'",
        "'a'+'λ'",
        "('a'+'λ')+'b'",
        "''+''",
        "'a'*0",
        "'λ'*3",
        "3*'λ'",
        "'λ'*true",
        "'λ'*false",
        "'λ'*3.0",
        "'λ'*-1",
        "'λ'*2.5",
        "'λ'*none",
        "'λ'*'3'",
        "'ab'*50000001",
        "''*18446744073709551615",
        "'a'~1",
        "1~'a'",
        "none~true",
        "(0.0/0.0)~'!'",
        "(1.0/0.0)~(-1.0/0.0)",
        "-0.0~0.0",
        "'a'+'b' == 'ab'",
        "'a'<'λ'",
        "'a'<=('a'+'')",
        "'λ' in ('a'+'λ')",
        "1 in 'a1b'",
        "true in 'TrueFalse'",
        "'' in ''",
        "'x' not in ''",
        "'a'+1",
        "'a'-1",
        "'a'/1",
        "'a'**1",
        "'a' in 1",
        "none < 'a'",
        "'a' < []",
        "[] > 'a'",
        "'é' in ('é'*3)",
    ] {
        compare(source);
    }
    for n in [22, 23, 4096] {
        compare(&format!("'{}'+'λ'", "a".repeat(n)));
    }
    let source = syntax("{{ 'a' 'λ' '\\n' }}");
    let expr = source.expressions().last().unwrap();
    let planned = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .prepare(expr)
        .unwrap();
    assert_eq!(planned.owner.recipes.len(), 0);
    assert_eq!(
        planned.requirements(),
        MaterializationRequirements::default()
    );
    let result = planned.materialize().unwrap();
    let Some(Value::Text(text)) = result.value() else {
        panic!("text");
    };
    assert!(text.segments().count() > 1);
    assert_eq!(text.segments().collect::<String>(), "aλ\n");
}

#[test]
fn direct_literal_lists_plan_comparison_format_and_membership_before_payload_fill() {
    for source in [
        "[]",
        "[1,true,none,'a' 'λ',1.5]",
        "[]==[]",
        "[1]==[true]",
        "[1]<[2]",
        "[true]<[1]",
        "[1,2]>[1]",
        "[1,'a']==[1,'a']",
        "1 in [1,2]",
        "3 not in [1,2]",
        "'a' in ['a','b']",
        "[1] in [1]",
        "not []",
        "not [1]",
        "[1] or []",
        "[] or [2]",
        "true and [2]",
        "'prefix'~[1,none,true,'a' 'λ']",
        "[1]~[2]",
        "[1] in '[1]'",
        "[1+2]",
        "[[1]]",
        "[not true]",
        "[1]+1",
        "[1]-1",
    ] {
        compare(source);
    }
    let source = syntax("{{ ('a'~[1,'é']) == ('b'~[2]) }}");
    let expr = source.expressions().last().unwrap();
    let planned = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .prepare(expr)
        .unwrap();
    assert!(planned.requirements().text_bytes() > 0);
    assert_eq!(planned.requirements().items(), 3);
    assert!(planned.owner.bytes.is_empty() && planned.owner.items.is_empty());
    assert!(matches!(
        planned.owner.value,
        Some(Descriptor::Scalar(Scalar::Bool(false)))
    ));
    let result = planned.materialize().unwrap();
    assert_eq!(result.owner.items.len(), 3);
    assert!(!result.owner.bytes.is_empty());
}

#[test]
fn materialized_list_nan_identity_and_scalar_order_use_distinct_eq_ord_rules() {
    let mut source = syntax("{{ [1.0] == [1.0] }}{{ [1.0] <= [1.0] }}");
    for record in &mut source.storage.expressions.nodes {
        if let R::Const(L::Float(x)) = &mut record.kind {
            *x = f64::from_bits(0x7ff8_0000_0000_1234);
        }
    }
    for expr in source.expressions() {
        let reference = project(&source, expr.id);
        let result = Plan::for_expression(expr)
            .unwrap()
            .construct()
            .unwrap()
            .fold(expr)
            .unwrap();
        exact(
            materialize(result.value()),
            crate::compiler::fold::materialization_reference::expression(&reference),
        );
    }
    let expr = source
        .expressions()
        .find(|e| matches!(source.storage.expressions.nodes[e.id.0].kind, R::List(_)))
        .unwrap();
    let planned = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .prepare(expr)
        .unwrap();
    let value = planned.owner.value.unwrap();
    assert!(planned.owner.equal(value, value).unwrap());
    assert!(planned.owner.compare_values(value, value).unwrap().is_eq());
}

#[test]
fn pure_default_scalar_and_joined_debug_bytes_match_untouched_display_debug() {
    use crate::compiler::fold::materialization_reference as old;
    for value in [
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        1e-7,
        1e20,
        1.0 / 0.0,
        -1.0 / 0.0,
        f64::from_bits(0x7ff8000000001234),
    ] {
        let ordinary = OrdinaryValue::from(value);
        let mut display = String::new();
        kernels::write_scalar(
            &mut display,
            crate::value::primitive::scalar::Scalar::F64(value),
            false,
        )
        .unwrap();
        assert_eq!(display, old::Display(&ordinary).to_string());
        let mut debug = String::new();
        kernels::write_scalar(
            &mut debug,
            crate::value::primitive::scalar::Scalar::F64(value),
            true,
        )
        .unwrap();
        assert_eq!(debug, format!("{ordinary:?}"));
    }
    for segments in [
        vec!["a", "\u{301}", "'\"\\\n"],
        vec!["\0\t\r", "中", "😀"],
        vec!["", ""],
    ] {
        let mut actual = String::new();
        kernels::write_debug_text(&mut actual, segments.iter().copied()).unwrap();
        assert_eq!(actual, format!("{:?}", segments.concat()));
    }
    for source in [
        "''~['a\\n' '\\t']",
        "''~['a' '\u{301}']",
        "''~['\\\"' \"'\"]",
    ] {
        compare(source);
    }
}

#[test]
fn actual_repeat_rule_and_integer_conversions_keep_boundaries_and_empty_fast_path() {
    assert_eq!(
        kernels::repeat_length(2, Some(50_000_000)),
        Ok((50_000_000, 100_000_000))
    );
    assert_eq!(
        kernels::repeat_length(2, Some(50_000_001)),
        Err(RepeatFailure::TooLarge)
    );
    assert_eq!(
        kernels::repeat_length(usize::MAX, Some(2)),
        Err(RepeatFailure::TooLarge)
    );
    assert_eq!(
        kernels::repeat_length(0, Some(usize::MAX)),
        Ok((usize::MAX, 0))
    );
    assert_eq!(kernels::repeat_length(0, None), Err(RepeatFailure::Integer));
    use crate::value::primitive::scalar::Scalar as S;
    for value in [
        S::None,
        S::Bool(true),
        S::Bool(false),
        S::I64(-1),
        S::U64(u64::MAX),
        S::U128(u128::MAX),
        S::I128(i128::MAX),
        S::F64(-0.0),
        S::F64(2.5),
        S::F64(f64::INFINITY),
        S::F64(i64::MAX as f64),
    ] {
        let ordinary = match value {
            S::None => OrdinaryValue::from(()),
            S::Bool(x) => x.into(),
            S::I64(x) => x.into(),
            S::U64(x) => x.into(),
            S::I128(x) => x.into(),
            S::U128(x) => x.into(),
            S::F64(x) => x.into(),
        };
        assert_eq!(
            kernels::fixed_usize(value),
            crate::compiler::fold::materialization_reference::as_usize(&ordinary)
        );
    }
}

#[test]
fn four_actual_reserve_sites_retain_exact_successful_prefixes_and_real_causes() {
    let source = syntax("{{ ('a'~[1,'é']) == ('b'~[2]) }}");
    let expr = source.expressions().last().unwrap();
    for (ordinal, site) in [
        Buffer::Frames,
        Buffer::Recipes,
        Buffer::Bytes,
        Buffer::Items,
    ]
    .into_iter()
    .enumerate()
    {
        let constructed = Plan::for_expression(expr)
            .unwrap()
            .construct_inner(Some(site));
        let failure = match constructed {
            Err(error) => error,
            Ok(workspace) => workspace.prepare(expr).unwrap().materialize().unwrap_err(),
        };
        assert!(matches!(failure.cause(), Cause::Reserve(_)));
        assert!(std::error::Error::source(&failure).is_some());
        assert!(std::ptr::eq(failure.owner.source, &source));
        let capacity = failure.capacities();
        assert!(capacity[..ordinal].iter().all(|x| *x > 0));
        assert!(capacity[ordinal..].iter().all(|x| *x == 0));
        assert!(failure.owner.bytes.is_empty() && failure.owner.items.is_empty());
        if ordinal >= 2 {
            assert!(!failure.owner.recipes.is_empty());
        }
        let mut workspace = failure.into_workspace();
        workspace.reserve_failure = None;
        // Metadata failures must retry construction, since no implicit metadata growth is allowed.
        if ordinal >= 2 {
            let completed = workspace.fold(expr).unwrap();
            exact(
                materialize(completed.value()),
                Some(OrdinaryValue::from(false)),
            );
        }
    }
}

#[test]
fn source_mismatch_keeps_generated_payloads_and_successful_reuse_keeps_real_allocations() {
    let source = syntax("{{ 'a'~[1,'é'] }}");
    let foreign = syntax("{{ 'a'~[1,'é'] }}");
    let expr = source.expressions().last().unwrap();
    let result = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .fold(expr)
        .unwrap();
    let capacities = result.owner.capacities();
    let pointers = (
        result.owner.frames.as_ptr(),
        result.owner.recipes.as_ptr(),
        result.owner.bytes.as_ptr(),
        result.owner.items.as_ptr(),
    );
    let bytes = result.owner.bytes.clone();
    let failure = result
        .into_workspace()
        .prepare(foreign.expressions().last().unwrap())
        .unwrap_err();
    assert!(matches!(failure.cause(), Cause::Source));
    assert_eq!(failure.capacities(), capacities);
    assert_eq!(failure.owner.bytes, bytes);
    let result = failure.into_workspace().fold(expr).unwrap();
    assert_eq!(result.owner.capacities(), capacities);
    assert_eq!(
        (
            result.owner.frames.as_ptr(),
            result.owner.recipes.as_ptr(),
            result.owner.bytes.as_ptr(),
            result.owner.items.as_ptr()
        ),
        pointers
    );
}

#[test]
fn metadata_and_payload_overflow_and_short_capacity_keep_pending_operation_prefix() {
    assert!(matches!(
        super::super::materialize::payload_layout(usize::MAX, 0),
        Err(Cause::Overflow)
    ));
    assert!(matches!(
        super::super::materialize::payload_layout(0, usize::MAX),
        Err(Cause::Overflow)
    ));
    let source = syntax("{{ ('a'+'b')+'c' }}");
    let expr = source.expressions().last().unwrap();
    let mut workspace = Plan::for_expression(expr).unwrap().construct().unwrap();
    workspace.requirements.recipes = 1;
    let failure = workspace.prepare(expr).unwrap_err();
    assert!(matches!(failure.cause(), Cause::Capacity));
    assert_eq!(failure.owner.recipes.len(), 1);
    assert!(failure.owner.pending_recipe.is_some() && failure.owner.operation.is_some());
    let mut workspace = failure.into_workspace();
    workspace.requirements.recipes = source.expression_count();
    let mut planned = workspace.prepare(expr).unwrap();
    assert!(matches!(planned.owner.fill(), Err(Cause::Capacity)));
    let result = planned.materialize().unwrap();
    exact(
        materialize(result.value()),
        Some(OrdinaryValue::from("abc")),
    );
}

#[test]
fn eager_nonconstant_and_later_map_refusal_keep_materialized_prefixes_without_new_engine() {
    for (input, mode) in [
        ("{{ x + ('a'~[1]) }}", 0),
        ("{{ ('a'~[1])+{} }}", 1),
        ("{{ x < ('a'~[1]) }}", 2),
        ("{{ x < ('a'~[1]) < 1 }}", 3),
    ] {
        let source = syntax(input);
        let expr = source.expressions().last().unwrap();
        let result = Plan::for_expression(expr)
            .unwrap()
            .construct()
            .unwrap()
            .prepare(expr);
        match (mode, result) {
            (0, Ok(plan)) => {
                assert!(plan.owner.value.is_none());
                assert!(!plan.owner.recipes.is_empty());
                assert!(plan.requirements().text_bytes() > 0);
            }
            (1, Err(error)) => {
                assert!(matches!(
                    error.cause(),
                    Cause::NeedsMaterialization(Collection::Map)
                ));
                assert!(!error.owner.recipes.is_empty());
            }
            (2, Ok(plan)) => {
                assert!(matches!(source.storage.expressions.nodes[expr.id.0].kind, R::Binary(..)));
                assert!(plan.owner.value.is_none());
                assert!(!plan.owner.recipes.is_empty());
                assert!(plan.requirements().text_bytes() > 0);
            }
            (3, Ok(plan)) => {
                assert!(matches!(source.storage.expressions.nodes[expr.id.0].kind, R::Compare(..)));
                assert!(plan.owner.value.is_none());
                assert!(plan.owner.recipes.is_empty());
            }
            _ => panic!("shared traversal mismatch"),
        }
    }
}

#[test]
fn actual_recipe_edges_ranges_and_append_totals_hold_on_reused_source_expressions() {
    let source = syntax("{{ ('a'~[1,'λ'])+('z'*4) }}{{ 'a' 'b' }}{{ 'λ'*0 }}{{ [] }}");
    for expr in source.expressions() {
        let planned = Plan::for_expression(expr)
            .unwrap()
            .construct()
            .unwrap()
            .prepare(expr)
            .unwrap();
        assert!(planned.owner.recipes.len() <= source.expression_count());
        let mut bytes = 0usize;
        let mut items = 0usize;
        for (index, recipe) in planned.owner.recipes.iter().enumerate() {
            use super::super::materialize::{Operation, TextKind};
            let earlier = |value: Descriptor| match value {
                Descriptor::Text(super::super::materialize::TextDescriptor {
                    kind: TextKind::Generated(id),
                    ..
                })
                | Descriptor::List(id) => assert!(id.0 < index),
                _ => (),
            };
            match recipe.operation {
                Operation::Add(a, b) => {
                    earlier(Descriptor::Text(a));
                    earlier(Descriptor::Text(b));
                }
                Operation::Repeat(a, _) => earlier(Descriptor::Text(a)),
                Operation::DisplayPair { left, right, .. } => {
                    earlier(left);
                    earlier(right);
                }
                Operation::DirectList { .. } => (),
            }
            let counter = if matches!(recipe.operation, Operation::DirectList { .. }) {
                &mut items
            } else {
                &mut bytes
            };
            assert_eq!(recipe.start, *counter);
            *counter += recipe.len;
        }
        assert!(items <= source.storage.expressions.operands.len());
        assert_eq!(
            (bytes, items),
            (
                planned.requirements().text_bytes(),
                planned.requirements().items()
            )
        );
        let result = planned.materialize().unwrap();
        assert_eq!(
            (result.owner.bytes.len(), result.owner.items.len()),
            (bytes, items)
        );
    }
}

#[test]
fn independent_materialization_owners_and_flat_recipe_drop_survive_external_unwind() {
    let source = syntax("{{ 'λ'~[1,'é'] }}");
    let expr = source.expressions().last().unwrap();
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..4 {
            threads.push(scope.spawn(|| {
                let result = Plan::for_expression(expr)
                    .unwrap()
                    .construct()
                    .unwrap()
                    .fold(expr)
                    .unwrap();
                exact(
                    materialize(result.value()),
                    Some(OrdinaryValue::from("λ[1, \"é\"]")),
                );
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
    });
    let input = format!("{{{{ {}'z' }}}}", "'a'+".repeat(200));
    let source = syntax(&input);
    let expr = source.expressions().last().unwrap();
    let result = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .fold(expr)
        .unwrap();
    assert_eq!(result.owner.recipes.len(), 200);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _owner = result;
        panic!("caller unwind");
    }));
    assert!(failure.is_err());
}

#[test]
fn ordinary_display_failure_chunks_and_dynamic_callbacks_keep_original_order_and_drop() {
    use crate::compiler::fold::materialization_reference as old;
    use crate::value::{ops, Object};
    use std::{
        fmt,
        sync::{Arc, Mutex},
    };
    struct Sink {
        writes: Vec<String>,
        fail: usize,
    }
    impl fmt::Write for Sink {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            self.writes.push(text.to_owned());
            if self.writes.len() == self.fail {
                Err(fmt::Error)
            } else {
                Ok(())
            }
        }
    }
    for value in [0.0, -0.0, f64::MAX, 1e-7, f64::NAN, f64::INFINITY] {
        let value = OrdinaryValue::from(value);
        for fail in 1..=4 {
            let mut actual = Sink {
                writes: Vec::new(),
                fail,
            };
            let mut expected = Sink {
                writes: Vec::new(),
                fail,
            };
            let a = write!(actual, "{value}");
            let b = write!(expected, "{}", old::Display(&value));
            assert_eq!(a, b);
            assert_eq!(actual.writes, expected.writes);
        }
    }
    #[derive(Debug)]
    struct Callback {
        log: Arc<Mutex<Vec<&'static str>>>,
        fails: bool,
    }
    impl Object for Callback {
        fn render(self: &Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.log.lock().unwrap().push("render");
            if self.fails {
                Err(fmt::Error)
            } else {
                f.write_str("callback")
            }
        }
    }
    impl Drop for Callback {
        fn drop(&mut self) {
            self.log.lock().unwrap().push("drop");
        }
    }
    for fails in [false, true] {
        for old_path in [false, true] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let value = OrdinaryValue::from_object(Callback {
                log: log.clone(),
                fails,
            });
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if old_path {
                    old::ops::string_concat(value, &OrdinaryValue::from("!"))
                } else {
                    ops::string_concat(value, &OrdinaryValue::from("!"))
                }
            }));
            assert_eq!(result.is_err(), fails);
            if let Ok(value) = result {
                assert_eq!(value.as_str(), Some("callback!"));
            }
            assert_eq!(*log.lock().unwrap(), ["render", "drop"]);
        }
    }
}
