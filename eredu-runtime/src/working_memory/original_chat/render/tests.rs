use super::super::tests::{consumer, messages, source_plan, tokenizer};
use super::*;
use eredu_text::chat_storage::ChatRenderBuffer;

#[test]
fn h_unwind_and_poison_keep_completed_or_partial_render_and_original_sources() {
    for partial in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let c = tokenizer(&pool);
        let j = pool.compile_chat_template(source_plan()).unwrap();
        let input = messages();
        let base = c.original_bytes() + j.original_bytes();
        let plan = j
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap();
        let h = required(&plan, &consumer()).unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = pool.render_original_chat_plan(&j, &c, plan, consumer(), || {
                    panic!("unwind after real H admission")
                });
            }))
            .is_err()
        );
        assert_eq!(pool.used_bytes().unwrap(), base);
        let plan = j
            .payload()
            .source
            .render_plan(ChatMessages::from_text(&input))
            .unwrap();
        let plan = if partial {
            plan.fail_reservation(ChatRenderBuffer::WithPrompt)
        } else {
            plan
        };
        let error = pool
            .render_original_chat_plan(&j, &c, plan, consumer(), || {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _usage = pool.0.usage.lock().unwrap();
                        panic!("poison actual H account");
                    }))
                    .is_err()
                );
            })
            .unwrap_err();
        assert!(matches!(
            error.accounting_failure(),
            Some(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(error.retained_bytes(), h);
        if partial {
            assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
            assert!(error.completed.is_none());
        } else {
            let rendered = error.completed.as_ref().unwrap();
            assert!(rendered.prompt(false).contains("Nonempty policy"));
            assert_eq!(rendered.generation_suffix(), "<|im_start|>assistant\n");
            assert!(error.render_failure().is_none());
        }
        assert!(error.template.as_ref().unwrap().same_source(&j));
        assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
        drop((j, c));
        drop(error);
        let usage = pool.0.usage.lock().unwrap_err().into_inner();
        assert_eq!(usage.reservations, 1);
        assert_eq!(
            usage.reserved,
            base + h,
            "poison does not fabricate a successful refund"
        );
    }
}

#[test]
fn borrowed_context_render_and_partial_failure_retain_only_original_owned_storage() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{{ prefix + suffix }}{% if add_generation_prompt %}!{% endif %}",
                "chat",
            )
            .unwrap(),
        )
        .unwrap();
    let base = pool.used_bytes().unwrap();
    let defaults = serde_json::json!({"prefix":"default", "suffix":"default"});
    let caller = serde_json::json!({"prefix":"borrowed 界", "suffix":" tail"});
    let context =
        ChatRenderContext::from_json(&[], defaults.as_object(), caller.as_object()).unwrap();
    let expected = pool
        .chat_render_required_bytes_with_context(&j, &c, context, consumer())
        .unwrap();
    let render = pool
        .render_original_chat_with_context(&j, &c, context, consumer())
        .unwrap();
    assert_eq!(render.original_bytes(), expected);
    assert_eq!(pool.used_bytes().unwrap(), base + expected);
    let alias = render.clone();
    drop((defaults, caller, render));
    assert_eq!(alias.prompt(true), "borrowed 界 tail!");
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), base);
    for buffer in [
        ChatRenderBuffer::Concat,
        ChatRenderBuffer::WithoutPrompt,
        ChatRenderBuffer::WithPrompt,
    ] {
        let caller = serde_json::json!({"prefix":"actual retained", "suffix":" prefix"});
        let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
        let plan = pool.chat_render_plan(&j, &c, context).unwrap();
        let expected = required(&plan, &consumer()).unwrap();
        let error = pool
            .render_original_chat_plan(&j, &c, plan.fail_reservation(buffer), consumer(), || {})
            .unwrap_err();
        drop(caller);
        assert_eq!(error.retained_bytes(), expected);
        assert!(error.render_failure().unwrap().retained_buffer_bytes() > 0);
        assert_eq!(pool.used_bytes().unwrap(), base + expected);
        drop(error);
        assert_eq!(pool.used_bytes().unwrap(), base);
    }
    drop((j, c));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn nested_borrowed_context_retains_new_text_destination_and_original_h_until_retirement() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8("{{ cfg.prefix + cfg.suffix }}", "chat").unwrap(),
        )
        .unwrap();
    let base = pool.used_bytes().unwrap();
    let caller = serde_json::json!({"cfg":{"prefix":"actual nested 界", "suffix":" copy"}});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let bytes = pool
        .chat_render_required_bytes_with_context(&j, &c, context, consumer())
        .unwrap();
    let render = pool
        .render_original_chat_with_context(&j, &c, context, consumer())
        .unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::ContextText);
    let failure = pool
        .render_original_chat_plan(&j, &c, plan, consumer(), || {})
        .unwrap_err();
    assert_eq!(failure.retained_bytes(), bytes);
    assert!(failure.render_failure().unwrap().retained_buffer_bytes() > 0);
    drop((caller, j, c));
    assert_eq!(pool.used_bytes().unwrap(), base + 2 * bytes);
    assert_eq!(render.prompt(true), "actual nested 界 copy");
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    drop(render);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn typed_short_circuit_chat_keeps_actual_render_and_failure_custody_after_context_retirement() {
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(r#"{% if cfg.reasoning is false or cfg.payload is mapping %}{{ (cfg.preferred or cfg.fallback).text + cfg.suffix }}{% endif %}{% if add_generation_prompt is true %}{{ cfg.suffix }}{% endif %}"#, "chat").unwrap(),
        )
        .unwrap();
    let base = pool.used_bytes().unwrap();
    let caller = serde_json::json!({"cfg":{"preferred":{},"fallback":{"text":"actual nested 界"},"suffix":" copy","reasoning":true,"payload":{"type":"text"}}});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let bytes = pool
        .chat_render_required_bytes_with_context(&j, &c, context, consumer())
        .unwrap();
    let render = pool
        .render_original_chat_with_context(&j, &c, context, consumer())
        .unwrap();
    let plan = pool
        .chat_render_plan(&j, &c, context)
        .unwrap()
        .fail_reservation(ChatRenderBuffer::ContextText);
    let failure = pool
        .render_original_chat_plan(&j, &c, plan, consumer(), || {})
        .unwrap_err();
    assert_eq!(failure.retained_bytes(), bytes);
    assert!(failure.render_failure().unwrap().retained_buffer_bytes() > 0);
    drop((caller, j, c));
    assert_eq!(pool.used_bytes().unwrap(), base + 2 * bytes);
    assert_eq!(render.prompt(false), "actual nested 界 copy");
    assert_eq!(render.prompt(true), "actual nested 界 copy copy");
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), base + bytes);
    drop(render);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_sized_measurement_is_admitted_before_nested_dispatch() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let template = "{% for call in calls %}{% for key,value in call.items() %}{{ key + ':' + value }};{% endfor %}{% endfor %}";
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "nested").unwrap())
        .unwrap();
    let base = pool.used_bytes().unwrap();
    let prefix = measurement_required(&j.payload().source).unwrap();
    let caller = serde_json::json!({"calls":[{"é":"one"},{"界":"two"}]});
    let context = ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap();
    let measured = pool
        .chat_render_required_bytes_with_context(&j, &c, context, consumer())
        .unwrap();
    assert_eq!(
        pool.used_bytes().unwrap(),
        base,
        "measurement scratch actually retired"
    );
    let rendered = pool
        .render_original_chat_with_context(&j, &c, context, consumer())
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), base + measured);
    drop((caller, j, c));
    assert_eq!(rendered.prompt(false), "é:one;界:two;");
    drop(rendered);
    assert_eq!(pool.used_bytes().unwrap(), 0);

    // Keep exactly the source owners and one byte less than real measurement.
    let pool = WorkingMemoryPool::new(base + prefix - 1, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(ChatTemplatePlan::prepare_utf8(template, "nested").unwrap())
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), base);
    let caller = serde_json::json!({"calls":[{"é":"one"}]});
    let error = pool
        .render_original_chat_with_context(
            &j,
            &c,
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
            consumer(),
        )
        .unwrap_err();
    assert!(error.accounting_failure().is_some());
    assert_eq!(
        error.retained_bytes(),
        0,
        "no scratch was constructed before refusal"
    );
    assert_eq!(pool.used_bytes().unwrap(), base);
}

#[test]
fn failed_loop_measurement_retains_its_source_and_paid_error_custody() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let c = tokenizer(&pool);
    let j = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_utf8(
                "{% for a,b in rows %}{{ a }}{{ b }}{% endfor %}",
                "bad-unpack",
            )
            .unwrap(),
        )
        .unwrap();
    let base = pool.used_bytes().unwrap();
    let prefix = measurement_required(&j.payload().source).unwrap();
    let caller = serde_json::json!({"rows":[["only one"]]});
    let error = pool
        .render_original_chat_with_context(
            &j,
            &c,
            ChatRenderContext::from_json(&[], None, caller.as_object()).unwrap(),
            consumer(),
        )
        .unwrap_err();
    assert_eq!(error.retained_bytes(), prefix);
    assert!(matches!(error.cause, Cause::Plan(_)));
    drop((caller, j, c));
    assert_eq!(pool.used_bytes().unwrap(), base + prefix);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}


#[test]
fn original_json_attempts_and_escaped_outputs_keep_exact_sources_and_h() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool=WorkingMemoryPool::new(u64::MAX,0).unwrap();
    let c=tokenizer(&pool);
    let j=pool.compile_chat_template(ChatTemplatePlan::prepare_utf8(
        "{{ tools|tojson(sort_keys=true,ensure_ascii=true) }}", "original-json").unwrap()).unwrap();
    let input=serde_json::json!({"tools":[{"z":7,"a":[{"z":-11,"a":{"z":23,"a":"É🙂"}}]}]});
    let context=ChatRenderContext::from_json(&[],None,input.as_object()).unwrap();
    let baseline=pool.used_bytes().unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap();
    assert_eq!(pool.used_bytes().unwrap(),baseline,"all temporary measurement scratch settled");
    let rendered=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap().fail_reservation(ChatRenderBuffer::ContextText);
    let error=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes()>0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes()+error.retained_bytes());
    drop((input,j,c));
    assert!(rendered.prompt(false).contains("23"));
    assert!(rendered.prompt(false).contains("\\ud83d\\ude42"));
    drop(error);assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes());
    drop(rendered);assert_eq!(pool.used_bytes().unwrap(),0);
}


#[test]
fn original_generated_slice_attempts_retire_scratch_and_retain_escaped_sources() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool=WorkingMemoryPool::new(u64::MAX,0).unwrap();
    let c=tokenizer(&pool);
    let j=pool.compile_chat_template(ChatTemplatePlan::prepare_utf8(
        "{% set out=payload|tojson %}{{ out[:-1] }}", "generated-slice").unwrap()).unwrap();
    let input=serde_json::json!({"payload":{"label":"É🙂","value":23}});
    let context=ChatRenderContext::from_json(&[],None,input.as_object()).unwrap();
    let baseline=pool.used_bytes().unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap();
    assert_eq!(pool.used_bytes().unwrap(),baseline,"every fixed prefix attempt retires its temporary buffers");
    let rendered=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap().fail_reservation(ChatRenderBuffer::ContextText);
    let error=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes()>0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes()+error.retained_bytes());
    drop((input,j,c));
    assert!(rendered.prompt(false).contains("23"));
    assert!(!rendered.prompt(false).ends_with('}'));
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes());
    drop(rendered);
    assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn original_generated_lists_retire_attempts_and_retain_failed_value_destinations() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let pool=WorkingMemoryPool::new(u64::MAX,0).unwrap();
    let c=tokenizer(&pool);
    let j=pool.compile_chat_template(ChatTemplatePlan::prepare_utf8(
        "{% set values=[payload.label,payload.value] %}{{ values[1] }}{{ values[0] }}", "generated-values").unwrap()).unwrap();
    let input=serde_json::json!({"payload":{"label":"É🙂","value":23}});
    let context=ChatRenderContext::from_json(&[],None,input.as_object()).unwrap();
    let baseline=pool.used_bytes().unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap();
    assert_eq!(pool.used_bytes().unwrap(),baseline,"every fixed prefix attempt retires its temporary buffers");
    let rendered=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap();
    let plan=pool.chat_render_plan(&j,&c,context).unwrap().fail_reservation(ChatRenderBuffer::Values);
    let error=pool.render_original_chat_plan(&j,&c,plan,consumer(),||{}).unwrap_err();
    assert!(error.render_failure().unwrap().retained_buffer_bytes()>0);
    assert!(error.template.as_ref().unwrap().same_source(&j));
    assert!(error.tokenizer.as_ref().unwrap().same_source(&c));
    assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes()+error.retained_bytes());
    drop((input,j,c));
    assert!(rendered.prompt(false).contains("23"));
    assert_eq!(rendered.prompt(false),"23É🙂");
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(),baseline+rendered.original_bytes());
    drop(rendered);
    assert_eq!(pool.used_bytes().unwrap(),0);
}


#[test]
fn amortized_measurement_falls_back_to_exact_reached_capacity_without_leaking_attempts() {
    use eredu_text::chat_storage::ChatTemplatePlan;
    let template = "{% for row in rows %}{{ (row + 'é')|upper }};{% endfor %}";
    let input=serde_json::json!({"rows":(0..33).map(|i|format!("row-{i}")).collect::<Vec<_>>()});
    let context=ChatRenderContext::from_json(&[],None,input.as_object()).unwrap();
    let pool=WorkingMemoryPool::new(u64::MAX,0).unwrap();
    let c=tokenizer(&pool);
    let j=pool.compile_chat_template(ChatTemplatePlan::prepare_utf8(template,"growth").unwrap()).unwrap();
    let baseline=pool.used_bytes().unwrap();
    let (mut json,mut text,mut values)=(ChatJsonCapacity::default(),ChatTextCapacity::default(),ChatValueCapacity::default());
    let mut attempts=0;
    let exact_requirements=loop {
        attempts+=1;
        assert!(attempts<1000,"finite reached storage");
        match j.payload().source.render_plan_attempt_with_values(context,json,text,values) {
            Ok(plan)=>break plan.requirements().required_bytes(),
            Err(ChatRenderPlanError::JsonCapacity(next))=>json=json.union(next),
            Err(ChatRenderPlanError::TextCapacity(next))=>text=text.union(next),
            Err(ChatRenderPlanError::ValueCapacity(next))=>values=values.union(next),
            Err(error)=>panic!("unexpected measurement failure: {error:?}"),
        }
    };
    assert!(attempts>8,"fixture must require repeated generated prefix growth");
    let minimum=measurement_required_with_capacities(&j.payload().source,json,text,values).unwrap();
    let rendered=pool.chat_render_plan(&j,&c,context).unwrap().render().unwrap();
    assert_eq!(rendered.prompt(false),(0..33).map(|i|format!("ROW-{i}É;")).collect::<String>());
    drop((rendered,j,c));
    assert_eq!(pool.used_bytes().unwrap(),0);

    for short in [false,true] {
        let pool=WorkingMemoryPool::new(baseline+minimum-u64::from(short),0).unwrap();
        let c=tokenizer(&pool);
        let j=pool.compile_chat_template(ChatTemplatePlan::prepare_utf8(template,"growth").unwrap()).unwrap();
        assert_eq!(pool.used_bytes().unwrap(),baseline);
        let result=pool.chat_render_plan(&j,&c,context);
        if short {
            let error=result.unwrap_err();
            assert!(matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded {required_bytes,available_bytes}) if *required_bytes==minimum && *available_bytes==minimum-1));
            assert_eq!(error.retained_bytes(),0,"no destination exists for the refused attempt");
        } else {
            assert_eq!(result.unwrap().requirements().required_bytes(),exact_requirements);
        }
        assert_eq!(pool.used_bytes().unwrap(),baseline,"every completed measurement actually retired");
        drop((j,c));
        assert_eq!(pool.used_bytes().unwrap(),0);
    }
}
