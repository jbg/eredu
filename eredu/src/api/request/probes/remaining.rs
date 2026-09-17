//! Fixed observations of the actual two tool-history renders and shared order.
use super::{Operations, Probe};
use crate::runtime::chat::{
    DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC, DEEPSEEK31_STRUCTURAL_JSON_TOOL_SPEC,
    KIMI_K2_NATIVE_TOOL_SPEC, LLAMA3_JSON_TOOL_SPEC, LLAMA4_JSON_TOOL_SPEC,
    MINISTRAL_JSON_LIST_TOOL_SPEC, MISTRAL_JSON_LIST_TOOL_SPEC, NEMOTRON_NANO_JSON_LIST_TOOL_SPEC,
    NEMOTRON_NANO_V2_JSON_LIST_TOOL_SPEC, QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING,
    QWEN_XML_TOOL_SPEC, QWEN3_XML_TOOL_SPEC,
    dialect::{DECLARATIVE_DIALECT, DialectParameters, FormatDialect},
    harmony::{GPT_OSS_HARMONY_PARAMETERS, HARMONY_DIALECT},
    lfm2::{LFM2_DIALECT, LFM2_PARAMETERS},
};
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Facts {
    kimi: bool,
    harmony: bool,
    deepseek: bool,
    deepseek1: bool,
    deepseek2: bool,
    lfm: bool,
    qwen: bool,
    tagged_mapping: bool,
    tagged_string: bool,
    json_xml: bool,
    mistral: bool,
    mistral_space: bool,
    mistral_compact: bool,
    nemotron: bool,
    llama: bool,
}
impl Facts {
    pub(crate) fn inspect(text: &str) -> Self {
        let has = |required: &[&str]| required.iter().all(|part| text.contains(part));
        Self {
            kimi: has(&["<|tool_calls_section_begin|>", "<|tool_call_begin|>", "<|tool_call_argument_begin|>", "<|tool_call_end|>", "<|tool_calls_section_end|>", "## Return of abc123456", "__eredu_tool_result_probe__"]),
            harmony: has(&["assistant to=functions.eredu_probe_7c91", "<|message|>", "<|call|>", "functions.eredu_probe_7c91 to=assistant", "__eredu_tool_result_probe__"]),
            deepseek: has(&["<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>", "<｜tool▁sep｜>", "eredu_probe_7c91", "<｜tool▁call▁end｜>", "<｜tool▁calls▁end｜>", "__eredu_tool_result_probe__"]),
            deepseek1: text.contains("<｜tool▁call▁begin｜>function<｜tool▁sep｜>eredu_probe_7c91\n```json\n"),
            deepseek2: text.contains("<｜tool▁call▁begin｜>eredu_probe_7c91<｜tool▁sep｜>"),
            lfm: has(&["<|tool_call_start|>", "eredu_probe_7c91(", "<|tool_call_end|>", "__eredu_tool_result_probe__"]),
            qwen: has(&["<tool_call>", "eredu_probe_7c91", "<tool_response>", "__eredu_tool_result_probe__"]),
            tagged_mapping: text.contains("<tool_call>\n<function=eredu_probe_7c91>\n<parameter=value>\n__eredu_mapping_argument_probe__\n</parameter>\n</function>\n</tool_call>"),
            tagged_string: has(&["<function=eredu_probe_7c91>\n<parameter=value>", "__eredu_string_argument_probe__"]),
            json_xml: has(&["\"name\": \"eredu_probe_7c91\"", "\"arguments\":"]),
            mistral: has(&["[TOOL_CALLS]", "eredu_probe_7c91", "abc123456", "[TOOL_RESULTS]", "__eredu_tool_result_probe__"]),
            mistral_space: text.contains("[TOOL_CALLS] ["),
            mistral_compact: text.contains("[TOOL_CALLS]["),
            nemotron: has(&["<TOOLCALL>[", "eredu_probe_7c91", "<TOOL_RESPONSE>[", "__eredu_tool_result_probe__"]),
            llama: has(&["eredu_probe_7c91", "\"parameters\"", "__eredu_tool_result_probe__"]),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Kimi,
    Harmony,
    DeepSeek1,
    DeepSeek2,
    Lfm,
    Qwen36,
    Qwen38,
    QwenReasoning,
    Xml,
    Mistral,
    Ministral,
    Nemotron,
    Nemotron2,
    Llama3,
    Llama4,
}
impl Kind {
    pub(crate) fn declaration(
        self,
    ) -> (&'static str, &'static dyn FormatDialect, DialectParameters) {
        use DialectParameters::{Custom as C, Declarative as D};
        match self {
            Self::Kimi => (
                "kimi-k2.native-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&KIMI_K2_NATIVE_TOOL_SPEC),
            ),
            Self::Harmony => (
                "harmony.channels.v1",
                &HARMONY_DIALECT,
                C(&GPT_OSS_HARMONY_PARAMETERS),
            ),
            Self::DeepSeek1 => (
                "deepseek.structural-json-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&DEEPSEEK_STRUCTURAL_JSON_TOOL_SPEC),
            ),
            Self::DeepSeek2 => (
                "deepseek.structural-json-tools.v2",
                &DECLARATIVE_DIALECT,
                D(&DEEPSEEK31_STRUCTURAL_JSON_TOOL_SPEC),
            ),
            Self::Lfm => ("lfm2.python-tools.v1", &LFM2_DIALECT, C(&LFM2_PARAMETERS)),
            Self::Qwen36 => (
                "qwen3.6.tagged-parameter-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING),
            ),
            Self::Qwen38 => (
                "qwen3.8.tagged-parameter-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&QWEN_TAGGED_TOOL_SPEC_GENERATED_REASONING),
            ),
            Self::QwenReasoning => (
                "qwen.xml-tools.reasoning.v1",
                &DECLARATIVE_DIALECT,
                D(&QWEN3_XML_TOOL_SPEC),
            ),
            Self::Xml => ("xml-tools.v1", &DECLARATIVE_DIALECT, D(&QWEN_XML_TOOL_SPEC)),
            Self::Mistral => (
                "mistral.json-list-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&MISTRAL_JSON_LIST_TOOL_SPEC),
            ),
            Self::Ministral => (
                "mistral.json-list-tools.compact.v1",
                &DECLARATIVE_DIALECT,
                D(&MINISTRAL_JSON_LIST_TOOL_SPEC),
            ),
            Self::Nemotron => (
                "nemotron.json-list-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&NEMOTRON_NANO_JSON_LIST_TOOL_SPEC),
            ),
            Self::Nemotron2 => (
                "nemotron.json-list-tools.reasoning.v1",
                &DECLARATIVE_DIALECT,
                D(&NEMOTRON_NANO_V2_JSON_LIST_TOOL_SPEC),
            ),
            Self::Llama3 => (
                "llama.json-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&LLAMA3_JSON_TOOL_SPEC),
            ),
            Self::Llama4 => (
                "llama.python-channel-tools.v1",
                &DECLARATIVE_DIALECT,
                D(&LLAMA4_JSON_TOOL_SPEC),
            ),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Recognition {
    pub(crate) kind: Kind,
    pub(crate) mapping: bool,
    pub(crate) string: bool,
}
pub(crate) trait ProtocolOperations: Operations {
    fn protocol_facts(&mut self, mapping: bool) -> Result<Option<Facts>, Self::Error>;
}
pub(crate) fn recognize<O: ProtocolOperations>(
    ops: &mut O,
) -> Result<Option<Recognition>, O::Error> {
    let mapping = ops.protocol_facts(true)?;
    let string = ops.protocol_facts(false)?;
    // Variant selection intentionally uses the first successful render, even
    // when only the other render satisfied a protocol's common predicates.
    let first = mapping.or(string);
    let m = mapping.unwrap_or_default();
    let s = string.unwrap_or_default();
    let result = if m.kimi || s.kimi {
        Recognition {
            kind: Kind::Kimi,
            mapping: m.kimi,
            string: s.kimi,
        }
    } else if m.harmony || s.harmony {
        Recognition {
            kind: Kind::Harmony,
            mapping: m.harmony,
            string: s.harmony,
        }
    } else if m.deepseek || s.deepseek {
        let Some(first) = first else {
            return Ok(None);
        };
        let kind = if first.deepseek1 {
            Kind::DeepSeek1
        } else if first.deepseek2 {
            Kind::DeepSeek2
        } else {
            return Ok(None);
        };
        Recognition {
            kind,
            mapping: m.deepseek,
            string: s.deepseek,
        }
    } else if m.lfm || s.lfm {
        Recognition {
            kind: Kind::Lfm,
            mapping: m.lfm,
            string: s.lfm,
        }
    } else if m.qwen || s.qwen {
        if m.tagged_mapping || s.tagged_string {
            let newer = ops.render_matches(
                Probe::QwenEffort,
                &["Reasoning effort is set to low."],
                None,
            )?;
            Recognition {
                kind: if newer { Kind::Qwen38 } else { Kind::Qwen36 },
                mapping: m.tagged_mapping,
                string: false,
            }
        } else {
            let mapping = m.qwen && m.json_xml;
            let string = s.qwen && s.json_xml;
            if !mapping && !string {
                return Ok(None);
            }
            let reasoning = ops.render_matches(
                Probe::ReasoningHistory,
                &["<think>\n__eredu_reasoning_probe__\n</think>\n\n__eredu_visible_probe__"],
                None,
            )?;
            Recognition {
                kind: if reasoning {
                    Kind::QwenReasoning
                } else {
                    Kind::Xml
                },
                mapping,
                string,
            }
        }
    } else if m.mistral || s.mistral {
        let Some(first) = first else {
            return Ok(None);
        };
        let kind = if first.mistral_space {
            Kind::Mistral
        } else if first.mistral_compact {
            Kind::Ministral
        } else {
            return Ok(None);
        };
        Recognition {
            kind,
            mapping: m.mistral,
            string: s.mistral,
        }
    } else if m.nemotron || s.nemotron {
        let newer = ops.structural(&["<SPECIAL_12>"])?;
        Recognition {
            kind: if newer {
                Kind::Nemotron2
            } else {
                Kind::Nemotron
            },
            mapping: m.nemotron,
            string: s.nemotron,
        }
    } else if m.llama || s.llama {
        let newer = ops.structural(&["<|python_start|>", "<|python_end|>", "<|eot|>"])?;
        Recognition {
            kind: if newer { Kind::Llama4 } else { Kind::Llama3 },
            mapping: m.llama,
            string: s.llama,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(result))
}
pub(crate) fn control_bytes<E>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<[Facts; 3]>(),
        size_of::<[Option<Facts>; 3]>(),
        size_of::<Recognition>(),
        size_of::<Kind>(),
        size_of::<Option<Recognition>>(),
        size_of::<Result<Option<Recognition>, E>>(),
        size_of::<Result<Option<Facts>, E>>(),
        size_of::<Result<bool, E>>(),
        size_of::<[bool; 4]>(),
        size_of::<(&str, &[&str], std::slice::Iter<'_, &str>)>(),
        size_of::<[&str; 7]>(),
        size_of::<[&str; 7]>(),
        size_of::<[&str; 5]>(),
        size_of::<[&str; 5]>(),
        size_of::<[&str; 4]>(),
        size_of::<[&str; 4]>(),
        size_of::<[&str; 4]>(),
        size_of::<[&str; 3]>(),
        size_of::<[&str; 2]>(),
        size_of::<[&str; 2]>(),
        size_of::<[&str; 3]>(),
        size_of::<[&str; 1]>(),
        size_of::<(&'static str, &'static dyn FormatDialect, DialectParameters)>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
