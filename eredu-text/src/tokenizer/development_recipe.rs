//! Development-only images of the ordinary chat compiler.
//!
//! This runs the existing allocating loader, normalizer, compiler, and variable
//! analysis. Its output is source evidence, never a construction or render bound.

use std::collections::BTreeMap;

use minijinja::machinery::{Instruction, Instructions, get_compiled_template};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    ChatTemplateIdentity, chat_environment, load_model_chat_template_from_str,
    normalize_chat_template, normalize_conditional_keyword_arguments, normalize_generation_blocks,
    selected_template_id,
};

/// Emits the actual ordinary compiler image selected from a full tokenizer config.
///
/// This development operation allocates freely and may initialize ordinary
/// globals/TLS. No original source owner, accounting fact, or renderer is created.
/// The image describes the pinned public MiniJinja compiler for diagnostics;
/// prepared chat always compiles source through the ordinary public API.
pub fn emit(
    config: &str,
    model_id: &str,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    let templates = load_model_chat_template_from_str(config)?.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing chat template")
    })?;
    let selected = templates.select(None)?;
    let name = selected_template_id(model_id, &selected);
    let raw = selected.template();
    let after_generation = normalize_generation_blocks(raw);
    let after_keywords = normalize_conditional_keyword_arguments(&after_generation);
    let normalized = normalize_chat_template(raw);
    let mut environment = chat_environment();
    environment.add_template_owned(name.clone(), normalized.clone())?;
    let template = environment.get_template(&name)?;
    let compiled = get_compiled_template(&template);
    let mut free_names = template
        .undeclared_variables(false)
        .into_iter()
        .collect::<Vec<_>>();
    free_names.sort();
    let mut blocks = BTreeMap::new();
    for (name, instructions) in &compiled.blocks {
        blocks.insert(*name, program(instructions)?);
    }
    Ok(json!({
        "schema": "eredu-ordinary-chat-compiler-image-v1",
        "upstream": {"crate": "minijinja", "version": "2.24.0"},
        "config_sha256": Sha256::digest(config.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        "selection": {
            "model_id": model_id,
            "template_name": name,
            "named_entry": match selected.identity() {
                ChatTemplateIdentity::Single => None,
                ChatTemplateIdentity::Named(name) => Some(name.as_str()),
            },
            "tools": "absent",
        },
        "normalization": {
            "selected_source": raw,
            "after_generation_blocks": after_generation,
            "after_conditional_keywords": after_keywords,
            "after_slice_compatibility": normalized,
            "compiled_source": compiled.instructions.source(),
            "unchanged": raw == normalized,
        },
        "settings": {
            "trim_blocks": environment.trim_blocks(),
            "lstrip_blocks": environment.lstrip_blocks(),
            "keep_trailing_newline": environment.keep_trailing_newline(),
            "undefined_behavior": format!("{:?}", environment.undefined_behavior()),
            "initial_auto_escape": format!("{:?}", compiled.initial_auto_escape),
            "syntax": "ordinary default",
            "unknown_method_callback": "minijinja_contrib::pycompat::unknown_method_callback",
            "tojson": "eredu_text::tokenizer::json::tojson",
            "dict": "eredu_text::tokenizer::python_dict",
        },
        "free_names": free_names,
        "root": program(&compiled.instructions)?,
        "blocks": blocks,
        "geometry_status": "compiler diagnostics without allocation guarantees",
        "accounting_authority": false,
    }))
}

fn program(instructions: &Instructions<'_>) -> Result<Value, serde_json::Error> {
    let mut records = Vec::new();
    let mut histogram = BTreeMap::<String, usize>::new();
    let mut pc = 0u32;
    while let Some(instruction) = instructions.get(pc) {
        let encoded = serde_json::to_value(instruction)?;
        let opcode = encoded["op"].as_str().expect("tagged upstream instruction");
        *histogram.entry(opcode.to_owned()).or_default() += 1;
        let constant_kind = match instruction {
            Instruction::LoadConst(value) => Some(format!("{:?}", value.kind())),
            _ => None,
        };
        records.push(json!({
            "pc": pc,
            "instruction": encoded,
            "line": instructions.get_line(pc),
            "span": instructions.get_span(pc),
            "constant_kind": constant_kind,
        }));
        pc = pc
            .checked_add(1)
            .expect("development instruction index overflow");
    }
    Ok(json!({
        "name": instructions.name(),
        "instruction_count": records.len(),
        "opcode_counts": histogram,
        "instructions": records,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(source: &str) -> String {
        json!({"chat_template": source}).to_string()
    }

    #[test]
    fn actual_nonempty_loop_image_has_sorted_free_names_and_locations() {
        let source = "{% for message in messages %}{{ message.role + ':' + message.content }}{% endfor %}{% if add_generation_prompt %}A{% endif %}";
        let result = emit(&config(source), "chat").unwrap();
        assert_eq!(
            result["free_names"],
            json!(["add_generation_prompt", "messages"])
        );
        assert_eq!(result["normalization"]["compiled_source"], source);
        assert_eq!(result["settings"]["initial_auto_escape"], "None");
        let instructions = result["root"]["instructions"].as_array().unwrap();
        assert!(!instructions.is_empty());
        assert_eq!(result["root"]["instruction_count"], instructions.len());
        assert!(
            instructions
                .iter()
                .any(|r| r["instruction"]["op"] == "PushLoop")
        );
        assert!(instructions.iter().all(|r| r["line"].as_u64().is_some()));
        for (pc, instruction) in instructions.iter().enumerate() {
            assert_eq!(instruction["pc"], pc);
        }
    }

    #[test]
    fn actual_name_selects_autoescape_and_named_default_without_tools() {
        let source = json!({"chat_template": [
            {"name":"tool_use", "template":"unused"},
            {"name":"default", "template":"{{ messages }}"}
        ]})
        .to_string();
        let result = emit(&source, "model").unwrap();
        assert_eq!(
            result["selection"]["template_name"],
            "model::chat_template::default"
        );
        assert_eq!(result["selection"]["named_entry"], "default");
        assert_eq!(result["normalization"]["selected_source"], "{{ messages }}");
        let html = emit(&config("{{ messages }}"), "model.html").unwrap();
        assert_eq!(html["settings"]["initial_auto_escape"], "Html");
    }

    #[test]
    fn wrapper_normalization_runs_before_the_actual_compiler() {
        let source = "{%- generation %}hello{%- endgeneration %}";
        let result = emit(&config(source), "chat").unwrap();
        assert_eq!(result["normalization"]["selected_source"], source);
        assert_eq!(
            result["normalization"]["compiled_source"],
            "{%- if true %}hello{%- endif %}"
        );
        assert_eq!(result["normalization"]["unchanged"], false);
        let slice = emit(&config("{{ 'abc'[::-1] }}"), "chat").unwrap();
        assert_eq!(
            slice["normalization"]["after_conditional_keywords"],
            "{{ 'abc'[::-1] }}"
        );
        assert_ne!(
            slice["normalization"]["after_conditional_keywords"],
            slice["normalization"]["compiled_source"]
        );
        assert_eq!(
            slice["normalization"]["after_slice_compatibility"],
            slice["normalization"]["compiled_source"]
        );
    }

    #[test]
    fn malformed_metadata_and_real_compiler_errors_are_preserved() {
        assert!(emit("{}", "chat").is_err());
        assert!(emit(r#"{"chat_template":[{"name":"default","template":"a"},{"name":"default","template":"b"}]}"#, "chat").is_err());
        let error = emit(&config("{% if %}"), "chat").unwrap_err();
        assert!(error.downcast_ref::<minijinja::Error>().is_some());
    }

    #[test]
    #[ignore = "root-only pinned full config emitter; output path must be fresh"]
    fn emit_pinned_full_config() {
        use std::io::Write;
        let input = std::env::var_os("EREDU_CHAT_RECIPE_INPUT").expect("pinned input path");
        let output = std::env::var_os("EREDU_CHAT_RECIPE_OUTPUT").expect("fresh output path");
        let model = std::env::var("EREDU_CHAT_RECIPE_MODEL").expect("actual template cache name");
        let config = std::fs::read_to_string(input).unwrap();
        let image = emit(&config, &model).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        serde_json::to_writer_pretty(&mut file, &image).unwrap();
        file.write_all(b"\n").unwrap();
    }
}
