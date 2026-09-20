use super::*;

#[test]
fn mapping_iterables_keep_order_types_and_list_index_semantics() {
    let template = "{{ map.keys()|length }}:{{ map.values()|length }}:{{ map.items()|length }}:{{ map|items|length }}|{{ map.keys() is iterable }}:{{ map.keys() is sequence }}:{{ map.keys()|list is sequence }}|{{ map.items()[0][0] == (map.items()|list)[0][0] }}:{{ map.keys()[-1] == (map.keys()|list)[-1] }}|{{ map.keys()|join(' / ') }}|{{ map.values()|join(' / ') }}|{{ (map.items()|list)[0][0] }}={{ (map.items()|list)[0][1] }}|{{ (map|items|list)[-1][-2] }}={{ (map|items|list)[-1][-1] }}|{{ (map|list)[-1] }}|{{ (map.values()|list)[99] is undefined }}|{% if map.items() %}nonempty{% else %}empty{% endif %}";
    for map in [
        json!({"z":"first","é\u{0}界":"🦀","a":false,"n":null,"f":2.5}),
        json!({"":"empty key"}),
    ] {
        let caller = json!({"map":map});
        compare(
            template,
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
    }
    let caller = json!({"map":{}});
    compare(
        "{{ map.keys()|length }}:{{ map.items()|list|length }}|{{ map.values()|join(',') }}|{{ (map.items()|list)[0] is undefined }}|{% if map.items() %}wrong{% else %}empty{% endif %}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
}

#[test]
fn mapping_keys_pairs_and_values_retain_exact_borrowed_input_and_paid_output() {
    // Nanbeige/Muse enumerate these actual argument maps. Pair iteration and
    // unpacking remain separate; this exercises the shared projection itself.
    let template = "{{ (tool_call.arguments|items|list)[0][0] + suffix }}:{{ (tool_call.arguments|items|list)[0][1].title }}|{{ tool_call.arguments.keys()|list|join(separator) }}|{{ needle in tool_call.arguments.keys() }}|{{ number in flat.values() }}|{{ (flat.values()|list)[0] }}{% if add_generation_prompt %}|{{ (tool_call.arguments.values()|list)[0].title }}{% endif %}";
    let caller = json!({"tool_call":{"arguments":{"é\u{0}界":{"title":"borrowed 🦀"},"later":{"title":"second"}}},"flat":{"first":3,"second":false},"suffix":"!","separator":"<>","needle":"é\u{0}界","number":3.0});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    let source = ChatTemplatePlan::prepare_utf8(template, "mapping-views")
        .unwrap()
        .compile()
        .unwrap();
    let plan = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
        )
        .unwrap();
    assert!(rendered.retained_buffer_bytes() <= plan.requirements().buffer_bytes());
    let failure = plan
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((source, caller));
    assert!(rendered.prompt(true).contains("é\u{0}界!"));
    assert!(rendered.prompt(true).ends_with("borrowed 🦀"));
    assert!(failure.retained_buffer_bytes() > 0);
}

#[test]
fn mapping_views_match_dynamic_calls_and_keep_short_circuit() {
    for template in [
        "{{ map.keys(1) }}",
        "{{ map.items(extra=true) }}",
        "{{ map|items(1) }}",
        "{{ map|list(1) }}",
    ] {
        compare_outcome(template, json!({"map":{}}).as_object().unwrap());
    }
    for template in [
        "{{ value.keys() }}",
        "{{ value.values() }}",
        "{{ value.items() }}",
        "{{ value|items }}",
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "receiver")
            .unwrap()
            .compile()
            .unwrap();
        for value in [
            json!(null),
            json!([]),
            json!(false),
            json!(42),
            json!("text"),
        ] {
            let caller = json!({"value":value});
            assert!(
                source
                    .render_plan_with_context(
                        ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap()
                    )
                    .unwrap()
                    .render()
                    .is_err()
            );
        }
    }
    compare(
        "{% if false and missing.items() %}wrong{% else %}skipped{% endif %}|{{ missing|list|length }}|{{ none|list|length }}",
        &[],
        &serde_json::Map::new(),
        &serde_json::Map::new(),
    );
    let rendered = compare(
        "{% for entry in args recursive %}{{ entry }}{% endfor %}",
        &[],
        &serde_json::Map::new(),
        json!({"args":[7,11]}).as_object().unwrap(),
    );
    assert_eq!(rendered.prompt(false), "711");
}

#[test]
fn dictionary_sort_preserves_case_ties_unicode_pairs_and_nested_generated_keys() {
    let template = "{% for key,value in values|dictsort %}{{ key }}={{ value|tojson }};{% endfor %}|{% for key,value in {('z'+'Z'):1,('a'+'A'):2,'AA':3}|dictsort %}{{ key }}={{ value }};{% endfor %}|{{ {}|dictsort|length }}";
    let caller = json!({"values":{"z":1,"AA":{"value":[1,2]},"aa":2,"aA":3,"é":4,"É":5,"a":null}});
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert!(rendered.prompt(false).ends_with("|aA=2;AA=3;zZ=1;|0"));
    assert!(rendered.retained_buffer_bytes() > 0);
}

#[test]
fn dictionary_sort_matches_independent_stable_order_at_heap_boundaries() {
    for length in [0usize, 1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 128] {
        let mut values = serde_json::Map::new();
        for index in (0..length).rev() {
            let key = if index % 2 == 0 {
                format!("k{:03}", index / 2)
            } else {
                format!("K{:03}", index / 2)
            };
            values.insert(key, json!(index));
        }
        values.insert("É".into(), json!(9));
        values.insert("é".into(), json!(-3));
        let mut keys: Vec<_> = values.keys().collect();
        keys.sort_by_key(|key| key.to_ascii_lowercase());
        let expected = keys
            .into_iter()
            .map(|key| format!("{key};"))
            .collect::<String>();
        let caller = json!({"values":values});
        let rendered = compare(
            "{% for key,value in values|dictsort %}{{ key }};{% endfor %}",
            &[],
            &serde_json::Map::new(),
            caller.as_object().unwrap(),
        );
        assert_eq!(rendered.prompt(false), expected);
    }
}

#[test]
fn structural_equality_preserves_nested_values_numeric_coercion_and_missing_keys() {
    let template = "{% for pair in pairs %}{{ pair[0]==pair[1] }}:{{ pair[0]!=pair[1] }}|{% endfor %}{{ {'a':absent}=={'b':absent} }}|{{ {'z':[7,{'é':2.5}],'a':true}==expected }}|{{ [1,2,3]==range(1,4) }}";
    let caller = json!({
        "pairs":[
            [{"a":[1,{"z":2.5}],"b":false},{"b":false,"a":[1.0,{"z":2.5}]}],
            [{"a":[1,{"z":2.5}]},{"a":[1,{"z":2.6}]}],
            [[1,2],[1,2,3]],[{},[]],["1",1],[null,false],
            [{"é":"界🙂"},{"é":"界🙂"}],[{"a":null},{"b":null}]
        ],
        "expected":{"a":true,"z":[7,{"é":2.5}]}
    });
    let rendered = compare(
        template,
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(
        rendered.prompt(false),
        "True:False|False:True|False:True|False:True|False:True|False:True|True:False|False:True|False|True|True"
    );
}
#[test]
fn structural_comparison_keeps_deep_borrowed_frames_and_short_circuits() {
    let mut left = json!({"n":7,"text":"É界🙂"});
    let mut right = left.clone();
    for _ in 0..48 {
        left = json!([left]);
        right = json!([right]);
    }
    let caller = json!({"left":left,"right":right});
    let rendered = compare(
        "{{ left==right }}|{{ left==[false] }}|{{ left==left }}",
        &[],
        &serde_json::Map::new(),
        caller.as_object().unwrap(),
    );
    drop(caller);
    assert_eq!(rendered.prompt(false), "True|False|True");
}
#[test]
fn ordinary_structural_comparison_preserves_unknown_map_lengths_and_object_identity() {
    use minijinja::value::{Enumerator, Object, ObjectRepr, Value};
    use std::sync::Arc;
    #[derive(Debug)]
    struct UnknownMap {
        extra: bool,
    }
    impl Object for UnknownMap {
        fn repr(self: &Arc<Self>) -> ObjectRepr {
            ObjectRepr::Map
        }
        fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
            None
        }
        fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
            match key.as_str()? {
                "a" => Some(Value::from(7)),
                "b" if self.extra => Some(Value::from(11)),
                _ => None,
            }
        }
        fn enumerate(self: &Arc<Self>) -> Enumerator {
            let extra = self.extra;
            Enumerator::Iter(Box::new(
                ["a", "b"]
                    .into_iter()
                    .take(if extra { 2 } else { 1 })
                    .map(Value::from),
            ))
        }
    }
    let a = Value::from_object(UnknownMap { extra: false });
    let same = Value::from_object(UnknownMap { extra: false });
    let extra = Value::from_object(UnknownMap { extra: true });
    assert_eq!(a, same);
    assert_ne!(a, extra);
    assert_ne!(extra, a);
    let nan = Value::from(vec![f64::NAN]);
    assert_eq!(nan, nan.clone());
    assert_ne!(nan, Value::from(vec![f64::NAN]));
}
