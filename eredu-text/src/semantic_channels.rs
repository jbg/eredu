//! Borrowed literal channel transitions shared by ordinary and prepared consumers.
//! Protocol selection, storage, event publication and tool payload parsing remain
//! with their owners. These descriptors grant no runtime source authority.
/// Semantic channel selected by an explicit source descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// Delimited reasoning output.
    Reasoning,
    /// Visible text output.
    Text,
}
/// Borrowed delimiters; this descriptor owns no allocation or source authority.
#[derive(Debug, Clone, Copy)]
pub struct DelimitedChannel<'a> {
    /// Opening delimiter.
    pub prefix: &'a str,
    /// Closing delimiter.
    pub suffix: &'a str,
    /// The reasoning opening delimiter was already emitted by the prompt.
    pub prefix_in_prompt: bool,
}
/// Source-selected literal channel program. No model/profile recognition occurs.
#[derive(Debug, Clone, Copy)]
pub struct ChannelProgram<'a> {
    /// Optional reasoning channel.
    pub reasoning_channel: Option<DelimitedChannel<'a>>,
    /// Optional visible-text channel.
    pub text_channel: Option<DelimitedChannel<'a>>,
    /// Exact handoff delimiter selected by the facade's actual protocol.
    pub tool_delimiter: &'a str,
    /// Keep the handoff delimiter as input to the existing JSON payload parser.
    pub tool_is_json: bool,
}
impl ChannelProgram<'_> {
    /// Checks progress of the finite channel traversal without allocating.
    /// Every execution consumer validates its immutable program before looping.
    pub fn is_valid(&self) -> bool {
        !self.tool_delimiter.is_empty()
            && [self.reasoning_channel, self.text_channel]
                .iter()
                .flatten()
                .all(|channel| !channel.prefix.is_empty() && !channel.suffix.is_empty())
    }
}
use std::mem::{size_of, size_of_val};
/// Borrowed cursor into the selected immutable delimiter program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State<'a> {
    /// Prompt already contains the reasoning opener; allow a tool handoff first.
    Prefilled {
        /// Output channel after prompt lookahead.
        kind: ChannelKind,
        /// Exact closing delimiter.
        suffix: &'a str,
    },
    /// Outside an explicitly delimited channel.
    Outside,
    /// Inside a delimited channel.
    Channel {
        /// Output channel.
        kind: ChannelKind,
        /// Exact closing delimiter.
        suffix: &'a str,
    },
}
impl<'a> State<'a> {
    /// Initial cursor determined solely by the supplied program.
    pub fn initial(spec: &ChannelProgram<'a>) -> Self {
        spec.reasoning_channel
            .filter(|channel| channel.prefix_in_prompt)
            .map_or(Self::Outside, |channel| Self::Prefilled {
                kind: ChannelKind::Reasoning,
                suffix: channel.suffix,
            })
    }
    /// Channel to receive pending bytes on ordinary end-of-stream.
    pub fn final_kind(self) -> ChannelKind {
        match self {
            Self::Outside => ChannelKind::Text,
            Self::Prefilled { kind, .. } | Self::Channel { kind, .. } => kind,
        }
    }
}
/// Next parser branch after consuming one descriptor.
#[derive(Debug, Clone, Copy)]
pub enum Next<'a> {
    /// Continue the literal channel worker with this cursor.
    Channel(State<'a>),
    /// Handoff to the selected tool payload parser.
    Tool,
}
/// One allocation-free emission and transition over borrowed pending bytes.
#[derive(Debug, Clone, Copy)]
pub struct Step<'a> {
    /// Output channel for the emitted prefix.
    pub kind: ChannelKind,
    /// UTF-8 byte length of the emitted prefix.
    pub emit: usize,
    /// UTF-8 byte length removed after emission succeeds.
    pub consume: usize,
    /// Optional state transition after consuming the prefix.
    pub next: Option<Next<'a>>,
    /// More input is required after this transition.
    pub wait: bool,
}
impl<'a> Step<'a> {
    fn waiting(kind: ChannelKind, emit: usize) -> Self {
        Self {
            kind,
            emit,
            consume: emit,
            next: None,
            wait: true,
        }
    }
    fn transition(next: State<'a>) -> Self {
        Self {
            kind: ChannelKind::Text,
            emit: 0,
            consume: 0,
            next: Some(Next::Channel(next)),
            wait: false,
        }
    }
}
#[derive(Clone, Copy)]
struct Delimiter<'a> {
    text: &'a str,
    next: Next<'a>,
}
fn tool_delimiter<'a>(spec: &ChannelProgram<'a>) -> &'a str {
    spec.tool_delimiter
}
fn tool_delimiter_is_json(spec: &ChannelProgram<'_>) -> bool {
    spec.tool_is_json
}
fn visible_prefix<'a>(pending: &str, delimiters: impl Iterator<Item = &'a str>) -> usize {
    let retained = delimiters
        .map(|delimiter| {
            (1..=delimiter.len().min(pending.len()))
                .rev()
                .find(|&length| {
                    let start = pending.len() - length;
                    pending.is_char_boundary(start) && delimiter.starts_with(&pending[start..])
                })
                .unwrap_or_default()
        })
        .max()
        .unwrap_or_default();
    pending.len() - retained
}
/// Produces one exact transition. Returned ranges are UTF-8 boundaries in
/// `pending`; equal-position delimiter ties retain ordinary declaration order.
pub fn step<'a>(spec: &ChannelProgram<'a>, state: State<'a>, pending: &str) -> Step<'a> {
    match state {
        State::Prefilled { kind, suffix } => {
            if pending.is_empty() {
                return Step::waiting(kind, 0);
            }
            let tool = tool_delimiter(spec);
            if pending.starts_with(tool) {
                Step::transition(State::Outside)
            } else if tool.starts_with(pending) {
                Step::waiting(kind, 0)
            } else {
                Step::transition(State::Channel { kind, suffix })
            }
        }
        State::Outside => {
            let delimiters = [
                spec.reasoning_channel
                    .filter(|channel| !channel.prefix_in_prompt)
                    .map(|channel| Delimiter {
                        text: channel.prefix,
                        next: Next::Channel(State::Channel {
                            kind: ChannelKind::Reasoning,
                            suffix: channel.suffix,
                        }),
                    }),
                spec.text_channel.map(|channel| Delimiter {
                    text: channel.prefix,
                    next: Next::Channel(State::Channel {
                        kind: ChannelKind::Text,
                        suffix: channel.suffix,
                    }),
                }),
                Some(Delimiter {
                    text: tool_delimiter(spec),
                    next: Next::Tool,
                }),
            ];
            let found = delimiters
                .iter()
                .flatten()
                .enumerate()
                .filter_map(|(index, delimiter)| {
                    pending
                        .find(delimiter.text)
                        .map(|position| (position, index, *delimiter))
                })
                .min_by_key(|(position, index, _)| (*position, *index));
            let Some((position, _, delimiter)) = found else {
                return Step::waiting(
                    ChannelKind::Text,
                    visible_prefix(pending, delimiters.iter().flatten().map(|d| d.text)),
                );
            };
            let skip = if matches!(delimiter.next, Next::Tool) && tool_delimiter_is_json(spec) {
                0
            } else {
                delimiter.text.len()
            };
            Step {
                kind: ChannelKind::Text,
                emit: position,
                consume: position + skip,
                next: Some(delimiter.next),
                wait: false,
            }
        }
        State::Channel { kind, suffix } => {
            if let Some(position) = pending.find(suffix) {
                Step {
                    kind,
                    emit: position,
                    consume: position + suffix.len(),
                    next: Some(Next::Channel(State::Outside)),
                    wait: false,
                }
            } else {
                Step::waiting(kind, visible_prefix(pending, std::iter::once(suffix)))
            }
        }
    }
}
/// Concrete stack facts for the shared worker, no payload or admission credit.
pub fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<State<'static>>(),
        size_of::<Step<'static>>(),
        size_of::<Next<'static>>(),
        size_of::<[Option<Delimiter<'static>>; 3]>(),
        size_of::<Option<(usize, usize, Delimiter<'static>)>>(),
        size_of::<(&ChannelProgram<'static>, State<'static>, &str)>(),
        size_of::<std::iter::Flatten<std::slice::Iter<'_, Option<Delimiter<'static>>>>>(),
        size_of::<std::iter::Rev<std::ops::RangeInclusive<usize>>>(),
        size_of::<Option<usize>>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<std::iter::Once<&str>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

mod tools;
pub use tools::{JsonEnvelope, JsonToolLayout, JsonToolProgram, JsonToolShape};

mod framing;
pub use framing::{JsonDelimiter, JsonFrame, JsonFrameError, JsonFrameFailure, JsonFrameStep, json_frame_control_bytes, json_frame_step};
