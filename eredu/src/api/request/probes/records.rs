//! One exact probe recipe; only the ordinary adapter creates serde containers.
use super::{Probe, ProbeInput};
use eredu_text::chat_storage::{
    ChatRecordField as F, ChatRecordValue as V, ChatScalarBinding as S,
    ChatScalarBindingValue as SV,
};

#[derive(Clone, Copy)]
pub(crate) enum Arguments<'a> {
    Mapping(&'a str),
    String(&'a str),
}
/// Lexical source frames, constructed only after the original caller has paid
/// the fixed recipe frames. No message or field references may escape the loan.
pub(crate) struct Input<'a> {
    pub(crate) messages: &'a [V<'a>],
    pub(crate) tools: &'a [V<'a>],
    pub(crate) kwargs: &'a [S<'a>],
    pub(crate) generation: bool,
}
impl Probe {
    pub(crate) fn with_records<T>(self, run: impl FnOnce(Input<'_>) -> T) -> T {
        match self {
            Self::Protocol { mapping } => tool(
                if mapping {
                    Arguments::Mapping("__eredu_mapping_argument_probe__")
                } else {
                    Arguments::String(r#"{"value":"__eredu_string_argument_probe__"}"#)
                },
                run,
            ),
            Self::QwenEffort => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_effort_probe__")),
                ];
                let messages = [V::Object(&user)];
                let kwargs = [S {
                    name: "reasoning_effort",
                    value: SV::Text("low"),
                }];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &kwargs,
                    generation: false,
                })
            }
            Self::IfmEffort(effort) => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_ifm_probe__")),
                ];
                let messages = [V::Object(&user)];
                let kwargs = [S {
                    name: "reasoning_effort",
                    value: SV::Text(effort.spelling()),
                }];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &kwargs,
                    generation: true,
                })
            }
            Self::IfmUser => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_ifm_probe__")),
                ];
                let messages = [V::Object(&user)];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &[],
                    generation: true,
                })
            }
            Self::IfmTools => ifm_tools(run),
            Self::InklingHistory => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_inkling_user_probe__")),
                ];
                let assistant = [
                    field("role", V::Text("assistant")),
                    field(
                        "reasoning_content",
                        V::Text("__eredu_inkling_reasoning_probe_7c91__"),
                    ),
                    field("content", V::Text("__eredu_inkling_visible_probe_28ad__")),
                ];
                let messages = [V::Object(&user), V::Object(&assistant)];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &[],
                    generation: false,
                })
            }
            Self::InklingEffort { enabled } => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_inkling_prompt_probe__")),
                ];
                let messages = [V::Object(&user)];
                let kwargs = [S {
                    name: "reasoning_effort",
                    value: SV::Text(if enabled { "high" } else { "none" }),
                }];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &kwargs,
                    generation: true,
                })
            }
            Self::GemmaHistory => gemma(true, true, run),
            Self::GemmaTool { mapping } => gemma(false, mapping, run),
            Self::GemmaPrompt(thinking) => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_prompt_probe__")),
                ];
                let messages = [V::Object(&user)];
                let kwargs = [S {
                    name: "enable_thinking",
                    value: SV::Bool(thinking),
                }];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &kwargs,
                    generation: true,
                })
            }
            Self::ReasoningHistory => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_user_probe__")),
                ];
                let assistant = [
                    field("role", V::Text("assistant")),
                    field("reasoning_content", V::Text("__eredu_reasoning_probe__")),
                    field("content", V::Text("__eredu_visible_probe__")),
                ];
                let messages = [V::Object(&user), V::Object(&assistant)];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &[],
                    generation: false,
                })
            }
            Self::MuseStrength(strength) => {
                let user = [
                    field("role", V::Text("user")),
                    field("content", V::Text("__eredu_atem_prompt__")),
                ];
                let messages = [V::Object(&user)];
                let kwargs = [S {
                    name: "reasoning_strength",
                    value: SV::Text(strength.spelling()),
                }];
                run(Input {
                    messages: &messages,
                    tools: &[],
                    kwargs: &kwargs,
                    generation: true,
                })
            }
            Self::Tool { mapping } => tool(
                if mapping {
                    Arguments::Mapping("probe-value")
                } else {
                    Arguments::String("{\"value\":\"probe-value\"}")
                },
                run,
            ),
        }
    }
}
fn field<'a>(key: &'a str, value: V<'a>) -> F<'a> {
    F { key, value }
}
pub(crate) fn tool<T>(arguments: Arguments<'_>, run: impl FnOnce(Input<'_>) -> T) -> T {
    let argument = [field(
        "value",
        V::Text(match arguments {
            Arguments::Mapping(value) => value,
            Arguments::String(_) => "",
        }),
    )];
    let argument = match arguments {
        Arguments::Mapping(_) => V::Object(&argument),
        Arguments::String(value) => V::Text(value),
    };
    let value_schema = [field("type", V::Text("string"))];
    let properties = [field("value", V::Object(&value_schema))];
    let required = [V::Text("value")];
    let parameters = [
        field("type", V::Text("object")),
        field("properties", V::Object(&properties)),
        field("required", V::Array(&required)),
        field("additionalProperties", V::Bool(false)),
    ];
    let definition = [
        field("name", V::Text("eredu_probe_7c91")),
        field("description", V::Text("protocol recognition probe")),
        field("parameters", V::Object(&parameters)),
    ];
    let tool = [
        field("type", V::Text("function")),
        field("function", V::Object(&definition)),
    ];
    let tools = [V::Object(&tool)];
    let user = [
        field("role", V::Text("user")),
        field("content", V::Text("__eredu_user_probe__")),
    ];
    let function = [
        field("name", V::Text("eredu_probe_7c91")),
        field("arguments", argument),
    ];
    let call = [
        field("id", V::Text("abc123456")),
        field("type", V::Text("function")),
        field("function", V::Object(&function)),
    ];
    let calls = [V::Object(&call)];
    let assistant = [
        field("role", V::Text("assistant")),
        field("content", V::Text("")),
        field("reasoning_content", V::Text("__eredu_reasoning_probe__")),
        field("thinking", V::Text("__eredu_reasoning_probe__")),
        field("tool_calls", V::Array(&calls)),
    ];
    let response = [
        field("role", V::Text("tool")),
        field("name", V::Text("eredu_probe_7c91")),
        field("tool_call_id", V::Text("abc123456")),
        field("content", V::Text("\"__eredu_tool_result_probe__\"")),
    ];
    let intermediate = [
        field("role", V::Text("assistant")),
        field("content", V::Text("__eredu_intermediate_assistant_probe__")),
    ];
    let followup = [
        field("role", V::Text("user")),
        field("content", V::Text("__eredu_followup_probe__")),
    ];
    let messages = [
        V::Object(&user),
        V::Object(&assistant),
        V::Object(&response),
        V::Object(&intermediate),
        V::Object(&followup),
    ];
    run(Input {
        messages: &messages,
        tools: &tools,
        kwargs: &[],
        generation: true,
    })
}
fn gemma<T>(history: bool, mapping: bool, run: impl FnOnce(Input<'_>) -> T) -> T {
    let value_schema = [field("type", V::Text("string"))];
    let properties = [field("value", V::Object(&value_schema))];
    let required = [V::Text("value")];
    let parameters = [
        field("type", V::Text("object")),
        field("properties", V::Object(&properties)),
        field("required", V::Array(&required)),
    ];
    let definition = [
        field("name", V::Text("eredu_probe")),
        field("description", V::Text("protocol probe")),
        field("parameters", V::Object(&parameters)),
    ];
    let tool = [
        field("type", V::Text("function")),
        field("function", V::Object(&definition)),
    ];
    let tools = [V::Object(&tool)];
    let user = [
        field("role", V::Text("user")),
        field(
            "content",
            V::Text(if history {
                "__eredu_user_probe__"
            } else {
                "probe"
            }),
        ),
    ];
    let args = [field(
        "value",
        V::Text(if history {
            "reasoning-probe"
        } else {
            "probe-value"
        }),
    )];
    let argument = if mapping {
        V::Object(&args)
    } else {
        V::Text("{\"value\":\"probe-value\"}")
    };
    let function = [
        field("name", V::Text("eredu_probe")),
        field("arguments", argument),
    ];
    let call = [
        field(
            "id",
            V::Text(if history {
                "reasoning-probe-call"
            } else {
                "probe-call"
            }),
        ),
        field("type", V::Text("function")),
        field("function", V::Object(&function)),
    ];
    let calls = [V::Object(&call)];
    // Separate field arrays preserve exact original insertion order and missing
    // reasoning_content in tool-only history (rather than an empty replacement).
    let history_assistant = [
        field("role", V::Text("assistant")),
        field(
            "reasoning_content",
            V::Text("__eredu_reasoning_probe_7c91__"),
        ),
        field("content", V::Text("__eredu_visible_probe_28ad__")),
        field("tool_calls", V::Array(&calls)),
    ];
    let tool_assistant = [
        field("role", V::Text("assistant")),
        field("content", V::Text("")),
        field("tool_calls", V::Array(&calls)),
    ];
    let response = [
        field("role", V::Text("tool")),
        field("tool_call_id", V::Text("probe-call")),
        field("content", V::Text("probe-response")),
    ];
    let messages = [
        V::Object(&user),
        V::Object(if history {
            &history_assistant
        } else {
            &tool_assistant
        }),
        V::Object(&response),
    ];
    let kwargs = [S {
        name: "enable_thinking",
        value: SV::Bool(true),
    }];
    run(Input {
        messages: &messages[..if history { 2 } else { 3 }],
        tools: &tools,
        kwargs: if history { &[] } else { &kwargs },
        generation: !history,
    })
}

fn ifm_tools<T>(run: impl FnOnce(Input<'_>) -> T) -> T {
    let value_schema = [field("type", V::Text("string"))];
    let properties = [field("value", V::Object(&value_schema))];
    let required = [V::Text("value")];
    let parameters = [
        field("type", V::Text("object")),
        field("properties", V::Object(&properties)),
        field("required", V::Array(&required)),
    ];
    let definition = [
        field("name", V::Text("eredu_ifm_probe")),
        field("description", V::Text("Probe")),
        field("parameters", V::Object(&parameters)),
    ];
    let tool = [
        field("type", V::Text("function")),
        field("function", V::Object(&definition)),
    ];
    let tools = [V::Object(&tool)];
    let user = [
        field("role", V::Text("user")),
        field("content", V::Text("__eredu_ifm_probe__")),
    ];
    let arguments = [field("value", V::Text("__eredu_value_probe__"))];
    let function = [
        field("name", V::Text("eredu_ifm_probe")),
        field("arguments", V::Object(&arguments)),
    ];
    let call = [
        field("type", V::Text("function")),
        field("id", V::Text("probe")),
        field("function", V::Object(&function)),
    ];
    let calls = [V::Object(&call)];
    let assistant = [
        field("role", V::Text("assistant")),
        field("think", V::Text("__eredu_think_probe__")),
        field("content", V::Text("")),
        field("tool_calls", V::Array(&calls)),
    ];
    let response = [
        field("role", V::Text("tool")),
        field("tool_call_id", V::Text("probe")),
        field("content", V::Text("__eredu_result_probe__")),
    ];
    let messages = [
        V::Object(&user),
        V::Object(&assistant),
        V::Object(&response),
    ];
    run(Input {
        messages: &messages,
        tools: &tools,
        kwargs: &[],
        generation: true,
    })
}

impl Input<'_> {
    /// Ordinary allocation policy over this same descriptor, never called by
    /// original preparation. Map insertion retains exact replacement order.
    pub(super) fn into_ordinary(self) -> ProbeInput {
        let kwargs = (!self.kwargs.is_empty()).then(|| {
            self.kwargs
                .iter()
                .map(|binding| {
                    let value = match binding.value {
                        SV::Bool(value) => serde_json::Value::Bool(value),
                        SV::Text(value) => serde_json::Value::String(value.into()),
                    };
                    (binding.name.into(), value)
                })
                .collect()
        });
        ProbeInput {
            messages: self.messages.iter().copied().map(ordinary_value).collect(),
            tools: self.tools.iter().copied().map(ordinary_value).collect(),
            kwargs,
            generation: self.generation,
        }
    }
}
fn ordinary_value(value: V<'_>) -> serde_json::Value {
    use serde_json::Value as J;
    match value {
        V::Null => J::Null,
        V::Bool(value) => J::Bool(value),
        V::Text(value) => J::String(value.into()),
        V::Array(values) => J::Array(values.iter().copied().map(ordinary_value).collect()),
        V::Object(values) => J::Object(
            values
                .iter()
                .map(|field| (field.key.into(), ordinary_value(field.value)))
                .collect(),
        ),
    }
}
/// Complete fixed descriptor populations of the three lexical workers. The
/// original caller pays these before entering the constructor callback; JSON
/// allocations above belong solely to the ordinary adapter.
pub(crate) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(
            [F<'_>; 1],
            [F<'_>; 1],
            [V<'_>; 1],
            [F<'_>; 3],
            [F<'_>; 3],
            [F<'_>; 2],
            [V<'_>; 1],
            [F<'_>; 2],
            [F<'_>; 1],
            V<'_>,
        )>(),
        size_of::<(
            [F<'_>; 2],
            [F<'_>; 3],
            [V<'_>; 1],
            [F<'_>; 4],
            [F<'_>; 3],
            [F<'_>; 3],
            [V<'_>; 3],
            [S<'_>; 1],
            bool,
            bool,
        )>(),
        size_of::<Input<'_>>(),
        size_of::<Probe>(),
        size_of::<Arguments<'_>>(),
        size_of::<([F<'_>; 2], [V<'_>; 1], [S<'_>; 1])>(),
        size_of::<([F<'_>; 2], [V<'_>; 1], [S<'_>; 1])>(),
        size_of::<([F<'_>; 2], [V<'_>; 1])>(),
        size_of::<(
            [F<'_>; 1],
            [F<'_>; 1],
            [V<'_>; 1],
            [F<'_>; 3],
            [F<'_>; 3],
            [F<'_>; 2],
            [V<'_>; 1],
        )>(),
        size_of::<(
            [F<'_>; 2],
            [F<'_>; 1],
            [F<'_>; 2],
            [F<'_>; 3],
            [V<'_>; 1],
            [F<'_>; 4],
            [F<'_>; 3],
            [V<'_>; 3],
        )>(),
        size_of::<([F<'_>; 2], [F<'_>; 3], [V<'_>; 2])>(),
        size_of::<([F<'_>; 2], [V<'_>; 1], [S<'_>; 1])>(),
        size_of::<(
            [F<'_>; 1],
            V<'_>,
            [F<'_>; 1],
            [F<'_>; 1],
            [V<'_>; 1],
            [F<'_>; 4],
            [F<'_>; 3],
            [F<'_>; 2],
            [V<'_>; 1],
        )>(),
        size_of::<(
            [F<'_>; 2],
            [F<'_>; 2],
            [F<'_>; 3],
            [V<'_>; 1],
            [F<'_>; 5],
            [F<'_>; 4],
            [F<'_>; 2],
            [F<'_>; 2],
            [V<'_>; 5],
        )>(),
        size_of::<(&str, V<'_>, F<'_>)>(),
        V::control_bytes()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
