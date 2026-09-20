use super::*;
use crate::chat_storage::ChatClockSnapshot;
#[test]
fn retained_clock_formats_original_calendar_values_and_paid_generated_text() {
    let date = chrono::NaiveDate::from_ymd_opt(2024, 2, 29)
        .unwrap()
        .and_hms_milli_opt(23, 59, 7, 321)
        .unwrap();
    let clock = ChatClockSnapshot::from_local(date);
    let base = [json!({"role":"user","content":"É界🙂"})];
    let caller = json!({"format":"%Y-%m-%d %a %b %j %G-W%V %H:%M:%S%.3f %% É界🙂","pieces":["%d ","%b ","%Y"]});
    let template = "{% macro date(format) %}{{ strftime_now(format) }}{% endmacro %}{% if strftime_now is defined %}{{ date(format) }}|{{ strftime_now(pieces|join('')) }}{% endif %}{% if add_generation_prompt %}|{{ strftime_now('%Y-%m-%d') }}{% endif %}";
    let source = ChatTemplatePlan::prepare_utf8(template, "clock")
        .unwrap()
        .compile()
        .unwrap();
    let context = ChatRenderContext::from_json(&base, None, caller.as_object())
        .unwrap()
        .with_clock(clock);
    let plan = source.render_plan_with_context(context).unwrap();
    let bound = plan.requirements().buffer_bytes();
    let rendered = plan.render().unwrap();
    assert!(rendered.retained_buffer_bytes() <= bound);
    let expected = format!(
        "{}|{}",
        date.format(caller["format"].as_str().unwrap()),
        date.format("%d %b %Y")
    );
    assert_eq!(rendered.prompt(false), expected);
    assert_eq!(rendered.prompt(true), format!("{expected}|2024-02-29"));
    let failure = source
        .render_plan_with_context(context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::WithPrompt)
        .render()
        .unwrap_err();
    drop((base, caller, source));
    assert!(failure.retained_buffer_bytes() > 0);
    assert!(rendered.prompt(true).ends_with("|2024-02-29"));
    drop((failure, rendered));
}
#[test]
fn clock_bindings_preserve_shadowing_and_reject_missing_offset_facts() {
    let date = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let clock = ChatClockSnapshot::from_local(date);
    for template in [
        "{{ strftime_now('%z') }}",
        "{{ strftime_now('%s') }}",
        "{{ strftime_now('%#z') }}",
        "{{ strftime_now(7) }}",
    ] {
        let source = ChatTemplatePlan::prepare_utf8(template, "clock")
            .unwrap()
            .compile()
            .unwrap();
        assert!(
            source
                .render_plan_with_context(
                    ChatRenderContext::from_messages(ChatMessages::from_text(&[]))
                        .with_clock(clock)
                )
                .unwrap()
                .render()
                .is_err()
        );
    }
    let vars = json!({"strftime_now":null});
    let source=ChatTemplatePlan::prepare_utf8("{% if strftime_now is defined %}defined{% endif %}:{% if strftime_now is none %}none{% endif %}","clock").unwrap().compile().unwrap();
    let rendered = source
        .render_plan_with_context(
            ChatRenderContext::from_json(&[], None, vars.as_object())
                .unwrap()
                .with_clock(clock),
        )
        .unwrap()
        .render()
        .unwrap();
    assert_eq!(rendered.prompt(false), "defined:none");
}
