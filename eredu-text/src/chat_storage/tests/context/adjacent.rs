use super::*;
#[test]
fn adjacent_loop_values_preserve_filtered_order_nested_macros_and_retained_borrows() {
    let values = json!({"groups":[{"rows":[{"keep":true,"text":"É"},{"keep":false,"text":"skip"},{"keep":true,"text":"界"},{"keep":true,"text":"🙂"}]},{"rows":[{"keep":true,"text":"only"}]}]});
    let template="{% macro show(value) %}{{ value.text|default('edge') }}{% endmacro %}{% for group in groups %}{% for row in group.rows if row.keep %}{% set before = loop.previtem %}{% set after = loop.nextitem %}{{ show(before) }}:{{ row.text }}:{{ show(after) }};{% endfor %}|{% endfor %}";
    // Accessing a member on undefined is an ordinary error; the macro guards
    // that actual boundary explicitly, as a released template does.
    let template = template.replace(
        "value.text|default('edge')",
        "value.text if value is defined else 'edge'",
    );
    let rendered = compare(
        &template,
        &[],
        &serde_json::Map::new(),
        values.as_object().unwrap(),
    );
    assert_eq!(
        rendered.prompt(false),
        "edge:É:界;É:界:🙂;界:🙂:edge;|edge:only:edge;|"
    );
    drop(values);
    assert!(rendered.prompt(true).contains("É:界:🙂"));
}
#[test]
fn safe_plain_text_uses_original_string_conversion_without_escaping() {
    let values = json!({"text":"<&> É界🙂","number":-2.5});
    let rendered = compare(
        "{{ text|safe }}|{{ number|safe }}|{{ absent|safe }}|{{ {'text':text}|tojson|safe }}",
        &[],
        &serde_json::Map::new(),
        values.as_object().unwrap(),
    );
    assert!(rendered.prompt(false).starts_with("<&> É界🙂|-2.5||"));
}
