//! Exact lexical boundaries shared by ordinary and prepared tool payloads.
//! These cursors do not replace complete JSON parsing or application schemas.
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Container {
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    String {
        escaped: bool,
    },
    Scalar,
}
/// Fixed lexical cursor for one JSON field value, after its first character.
#[derive(Clone, Copy, Debug)]
pub struct ValueCursor {
    kind: Kind,
    complete: bool,
}
/// A field delimiter occurred where the value's first character was required.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("declarative JSON field is missing a value")]
pub struct MissingValue;
/// The next character's exact effect on the value boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueStep {
    /// Append this character; `complete` closes a container or quoted value.
    Consumed {
        /// Whether the current value is lexically closed.
        complete: bool,
    },
    /// Leave this delimiter/whitespace for the enclosing object parser.
    Boundary,
}
impl ValueCursor {
    /// Classifies the already-consumed first character without allocating.
    pub fn start(character: char) -> Result<Self, MissingValue> {
        let kind = match character {
            '{' | '[' => Kind::Container {
                depth: 1,
                in_string: false,
                escaped: false,
            },
            '"' => Kind::String { escaped: false },
            ',' | '}' | ']' => return Err(MissingValue),
            _ => Kind::Scalar,
        };
        Ok(Self {
            kind,
            complete: false,
        })
    }
    /// Previews one exact transition. Publish the returned cursor only after
    /// the caller has appended a `Consumed` character to its actual storage.
    pub fn step(mut self, character: char) -> (Self, ValueStep) {
        if self.complete {
            return (self, ValueStep::Boundary);
        }
        let step = match &mut self.kind {
            Kind::Container {
                depth,
                in_string,
                escaped,
            } => {
                if *in_string {
                    if *escaped {
                        *escaped = false;
                    } else if character == '\\' {
                        *escaped = true;
                    } else if character == '"' {
                        *in_string = false;
                    }
                } else {
                    match character {
                        '"' => *in_string = true,
                        '{' | '[' => *depth += 1,
                        '}' | ']' => *depth -= 1,
                        _ => {}
                    }
                }
                ValueStep::Consumed {
                    complete: *depth == 0,
                }
            }
            Kind::String { escaped } => {
                let complete = if *escaped {
                    *escaped = false;
                    false
                } else if character == '\\' {
                    *escaped = true;
                    false
                } else {
                    character == '"'
                };
                ValueStep::Consumed { complete }
            }
            Kind::Scalar => {
                if character.is_whitespace() || matches!(character, ',' | '}' | ']') {
                    ValueStep::Boundary
                } else {
                    ValueStep::Consumed { complete: false }
                }
            }
        };
        if let ValueStep::Consumed { complete } = step {
            self.complete = complete;
        }
        (self, step)
    }
    /// Fixed cursor, character and constructor/transition result transports.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Kind>(),
            size_of::<char>(),
            size_of::<ValueStep>(),
            size_of::<MissingValue>(),
            size_of::<Result<Self, MissingValue>>(),
            size_of::<(Self, ValueStep)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
/// Fixed scanner for one complete JSON container, including leading whitespace.
#[derive(Clone, Copy, Debug, Default)]
pub struct FragmentCursor {
    value: Option<ValueCursor>,
    complete: bool,
}
/// A lexical refusal or the concrete failure from the caller's append worker.
#[derive(Debug, thiserror::Error)]
pub enum FragmentError<E> {
    /// The first non-whitespace character was neither `{` nor `[`.
    #[error("JSON fragment must start with a container")]
    MissingContainer,
    /// New input was supplied after the complete container had been consumed.
    #[error("JSON fragment has trailing data")]
    TrailingData,
    /// The exact caller-owned character destination refused the append.
    #[error("{0}")]
    Append(#[source] E),
}
impl FragmentCursor {
    /// Whether one complete container has already been consumed.
    pub fn is_complete(&self) -> bool {
        self.complete
    }
    /// Uses the actual shared character traversal and caller-owned destination.
    /// A failed append leaves the cursor at the last successful character.
    pub fn push<E, F>(
        &mut self,
        input: &str,
        mut append: F,
    ) -> Result<(usize, bool), FragmentError<E>>
    where
        F: FnMut(char) -> Result<(), E>,
    {
        if self.complete {
            return if input.is_empty() {
                Ok((0, true))
            } else {
                Err(FragmentError::TrailingData)
            };
        }
        let mut consumed = 0;
        for character in input.chars() {
            let next = match self.value {
                None if character.is_whitespace() => None,
                None if matches!(character, '{' | '[') => Some((
                    ValueCursor::start(character).expect("container start"),
                    false,
                )),
                None => return Err(FragmentError::MissingContainer),
                Some(value) => {
                    let (next, step) = value.step(character);
                    let ValueStep::Consumed { complete } = step else {
                        unreachable!("container cursor")
                    };
                    Some((next, complete))
                }
            };
            append(character).map_err(FragmentError::Append)?;
            consumed += character.len_utf8();
            if let Some((value, complete)) = next {
                self.value = Some(value);
                self.complete = complete;
                if complete {
                    break;
                }
            }
        }
        Ok((consumed, self.complete))
    }
    /// Exact fixed traversal and concrete callback/result controls. Buffer
    /// storage and callback work remain separately paid by the storage owner.
    pub fn control_bytes<E, F>(_: &F) -> Option<usize>
    where
        F: FnMut(char) -> Result<(), E>,
    {
        let parts = [
            ValueCursor::control_bytes()?,
            size_of::<Self>(),
            size_of::<Option<ValueCursor>>(),
            size_of::<Option<(ValueCursor, bool)>>(),
            size_of::<F>(),
            size_of::<E>(),
            size_of::<FragmentError<E>>(),
            size_of::<Result<(), E>>(),
            size_of::<Result<(usize, bool), FragmentError<E>>>(),
            size_of::<(&mut Self, &str, F)>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<(usize, char, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_container_scanner_preserves_split_bytes_and_failed_append_cursor() {
        let input = r#" {"a":["é","]}",17]}tail"#;
        let end = input.find("tail").unwrap();
        for split in (0..=end).filter(|&n| input.is_char_boundary(n)) {
            let mut cursor = FragmentCursor::default();
            let mut output = String::new();
            let (first, complete) = cursor
                .push(&input[..split], |c| {
                    output.push(c);
                    Ok::<_, &'static str>(())
                })
                .unwrap();
            assert_eq!(first, split);
            assert_eq!(complete, split == end);
            if !complete {
                let (rest, complete) = cursor
                    .push(&input[split..], |c| {
                        output.push(c);
                        Ok::<_, &'static str>(())
                    })
                    .unwrap();
                assert_eq!(rest, end - split);
                assert!(complete);
            }
            assert_eq!(output, input[..end]);
        }
        let mut cursor = FragmentCursor::default();
        let mut output = String::new();
        let stopped = cursor.push(input, |c| {
            if c == 'é' {
                return Err("destination");
            }
            output.push(c);
            Ok(())
        });
        assert!(matches!(stopped, Err(FragmentError::Append("destination"))));
        assert_eq!(output, &input[..input.find('é').unwrap()]);
        assert!(!cursor.is_complete());
        let consumed = output.len();
        let (count, complete) = cursor
            .push(&input[consumed..], |c| {
                output.push(c);
                Ok::<_, &'static str>(())
            })
            .unwrap();
        assert_eq!(count, end - consumed);
        assert!(complete);
        assert_eq!(output, input[..end]);
        assert!(matches!(
            cursor.push("x", |_| Ok::<_, &'static str>(())),
            Err(FragmentError::TrailingData)
        ));
    }
}

mod object;
pub use object::{ObjectContext, ObjectCursor, ObjectSyntaxError};

mod fields;
pub use fields::{JsonCallId, JsonFieldError, JsonFieldNames, JsonFieldRole, JsonValueKind};

mod call_id;
pub use call_id::{GENERATED_CALL_ID_BYTES, generated_call_id_control_bytes, write_generated_call_id};
