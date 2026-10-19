//! Host rendering of NVIDIA's VoiceChat function-calling system template.
//!
//! The wire text follows `examples/speechlm2/function_calling/template.jinja`
//! at NVIDIA-NeMo/Speech revision `097dfe9e2f55baf653b83035868bdc89849f1b47`.
//! Plain Jinja's JSON filter sorts keys, escapes Unicode and HTML-sensitive
//! characters, and uses Python's default separator spacing.

use std::fmt::{self, Write};

use eredu_text::tokenizer::write_python_json;
use serde_json::Value;

use crate::runtime::chat::tool_schema::parse_tools;

#[derive(Debug, thiserror::Error)]
pub(super) enum PromptError {
    #[error("rendered VoiceChat system prompt exceeds {capacity} bytes")]
    Capacity { capacity: usize },
    #[error("invalid VoiceChat tools: {0}")]
    Tools(String),
}

pub(super) fn render_system_prompt(
    system: &str,
    tools: &[Value],
    max_bytes: usize,
) -> Result<String, PromptError> {
    parse_tools(tools).map_err(PromptError::Tools)?;
    let mut output = BoundedText {
        text: String::new(),
        capacity: max_bytes,
    };
    render(&mut output, system, tools).map_err(|_| PromptError::Capacity {
        capacity: max_bytes,
    })?;
    Ok(output.text)
}

fn render(output: &mut BoundedText, system: &str, tools: &[Value]) -> fmt::Result {
    output.write_str(system)?;
    if tools.is_empty() {
        return Ok(());
    }
    if !system.is_empty() {
        output.write_str("\n\n")?;
    }
    output.write_str(
        "You can use the following tools to assist the user if required:\n<AVAILABLE_TOOLS>[",
    )?;
    for (index, tool) in tools.iter().enumerate() {
        if index != 0 {
            output.write_str(", ")?;
        }
        // Shared validation admits the facade's outer function object and
        // rejects an inner `type` field, which the reference template removes.
        write_python_json(&tool["function"], &mut HtmlSafe(output), true, true)?;
    }
    output.write_str(concat!(
        "]</AVAILABLE_TOOLS>\n\n",
        "If you decide to call any tool(s), use the following format:\n",
        "<TOOLCALL>[{\"name\": \"tool_name1\", \"arguments\": \"tool_args1\"}, ",
        "{\"name\": \"tool_name2\", \"arguments\": \"tool_args2\"}]</TOOLCALL>\n\n",
        "The user will execute tool-calls and return responses from tool(s) in this format:\n",
        "<TOOL_RESPONSE>[{\"tool_response1\"}, {\"tool_response2\"}]</TOOL_RESPONSE>\n\n",
        "Based on the tool responses, you can call additional tools if needed, ",
        "correct tool calls if any errors are found, or just respond to the user.",
    ))
}

struct BoundedText {
    text: String,
    capacity: usize,
}

impl Write for BoundedText {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if text.len() > self.capacity - self.text.len() {
            return Err(fmt::Error);
        }
        self.text.push_str(text);
        Ok(())
    }
}

struct HtmlSafe<'a>(&'a mut BoundedText);

impl Write for HtmlSafe<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        // Escape the serialized JSON bytes, exactly as Jinja's htmlsafe_json_dumps.
        for part in text.split_inclusive(['<', '>', '&', '\'']) {
            let (prefix, escaped) = match part.as_bytes().last() {
                Some(b'<') => (&part[..part.len() - 1], "\\u003c"),
                Some(b'>') => (&part[..part.len() - 1], "\\u003e"),
                Some(b'&') => (&part[..part.len() - 1], "\\u0026"),
                Some(b'\'') => (&part[..part.len() - 1], "\\u0027"),
                _ => (part, ""),
            };
            self.0.write_str(prefix)?;
            self.0.write_str(escaped)?;
        }
        Ok(())
    }
}
