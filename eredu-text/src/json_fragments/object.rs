//! Shared incremental object-field traversal over each caller's actual raw bytes.
use super::{ValueCursor, ValueStep};
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

/// Fixed syntax failures from the ordinary declarative tool object worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ObjectSyntaxError {
    /// The source did not start with a JSON object after whitespace.
    #[error("declarative tool call must be a JSON object")]
    ExpectedObject,
    /// An object key or allowed empty-object terminator was required.
    #[error("declarative tool-call JSON expected an object field")]
    ExpectedField,
    /// The completed key was not followed by a colon.
    #[error("declarative tool-call JSON expected ':' after a field")]
    ExpectedColon,
    /// A field was followed by neither comma nor object terminator.
    #[error("declarative tool-call JSON expected ',' or '}}' after a field")]
    ExpectedSeparator,
    /// A value was missing after its key and colon.
    #[error("declarative JSON field is missing a value")]
    MissingValue,
}
/// Actual destination and field-validation operations for one object cursor.
/// Implementations retain source identity and admission; this trait supplies none.
pub trait ObjectContext {
    /// Caller-owned decoded field identity.
    type Key;
    /// Concrete syntax/storage/validation failure.
    type Error;
    /// Current complete raw fragment length.
    fn raw_len(&self) -> usize;
    /// Append exactly one character to the actual fragment destination.
    fn append(&mut self, character: char) -> Result<(), Self::Error>;
    /// Decode and register this quoted raw key, rejecting actual duplicates.
    fn key(&mut self, raw: Range<usize>) -> Result<Self::Key, Self::Error>;
    /// Validate and publish this exact complete field from the raw destination.
    fn value(&mut self, key: Self::Key, raw: Range<usize>) -> Result<(), Self::Error>;
    /// Translate a fixed scanner refusal into the caller's normal typed error.
    fn syntax(&self, cause: ObjectSyntaxError) -> Self::Error;
}
#[derive(Clone, Debug)]
enum Phase<K> {
    Start,
    KeyOrEnd {
        allow_end: bool,
    },
    Key {
        start: usize,
        escaped: bool,
    },
    Colon {
        key: K,
    },
    ValueStart {
        key: K,
    },
    Value {
        key: K,
        start: usize,
        cursor: ValueCursor,
    },
    AfterValue,
}
/// Fixed transition state plus at most one caller-owned decoded key.
#[derive(Clone, Debug)]
pub struct ObjectCursor<K> {
    phase: Phase<K>,
    complete: bool,
}
impl<K> Default for ObjectCursor<K> {
    fn default() -> Self {
        Self {
            phase: Phase::Start,
            complete: false,
        }
    }
}
impl<K> ObjectCursor<K> {
    /// The current key, if a decoded key is held by this cursor.
    pub fn key(&self) -> Option<&K> {
        match &self.phase {
            Phase::Colon { key } | Phase::ValueStart { key } | Phase::Value { key, .. } => {
                Some(key)
            }
            _ => None,
        }
    }
    /// Raw span of the currently growing field, before its completion callback.
    pub fn active_value(&self, raw_end: usize) -> Option<(&K, Range<usize>)> {
        match &self.phase {
            Phase::Value { key, start, .. } => Some((key, *start..raw_end)),
            _ => None,
        }
    }
    fn finish<C: ObjectContext<Key = K>>(&mut self, context: &mut C) -> Result<(), C::Error> {
        let Phase::Value { key, start, .. } = std::mem::replace(&mut self.phase, Phase::AfterValue)
        else {
            unreachable!("only a field value can complete")
        };
        let end = context.raw_len();
        context.value(key, start..end)
    }
    /// Advance the exact ordinary object loop. Scalar boundary characters stay
    /// unconsumed until the enclosing field has completed successfully.
    pub fn push<C: ObjectContext<Key = K>>(
        &mut self,
        input: &str,
        context: &mut C,
    ) -> Result<(usize, bool), C::Error> {
        let mut consumed = 0;
        while consumed < input.len() && !self.complete {
            let character = input[consumed..].chars().next().expect("input character");
            let length = character.len_utf8();
            match &mut self.phase {
                Phase::Start => {
                    if !character.is_whitespace() && character != '{' {
                        return Err(context.syntax(ObjectSyntaxError::ExpectedObject));
                    }
                    context.append(character)?;
                    consumed += length;
                    if character == '{' {
                        self.phase = Phase::KeyOrEnd { allow_end: true };
                    }
                }
                Phase::KeyOrEnd { allow_end } => {
                    if character.is_whitespace() {
                        context.append(character)?;
                        consumed += length;
                    } else if character == '"' {
                        let start = context.raw_len();
                        context.append(character)?;
                        consumed += length;
                        self.phase = Phase::Key {
                            start,
                            escaped: false,
                        };
                    } else if character == '}' && *allow_end {
                        context.append(character)?;
                        consumed += length;
                        self.complete = true;
                    } else {
                        return Err(context.syntax(ObjectSyntaxError::ExpectedField));
                    }
                }
                Phase::Key { start, escaped } => {
                    context.append(character)?;
                    consumed += length;
                    if *escaped {
                        *escaped = false;
                    } else if character == '\\' {
                        *escaped = true;
                    } else if character == '"' {
                        let raw = *start..context.raw_len();
                        self.phase = Phase::Colon {
                            key: context.key(raw)?,
                        };
                    }
                }
                Phase::Colon { .. } => {
                    if character.is_whitespace() {
                        context.append(character)?;
                        consumed += length;
                    } else if character == ':' {
                        context.append(character)?;
                        consumed += length;
                        let Phase::Colon { key } =
                            std::mem::replace(&mut self.phase, Phase::AfterValue)
                        else {
                            unreachable!()
                        };
                        self.phase = Phase::ValueStart { key };
                    } else {
                        return Err(context.syntax(ObjectSyntaxError::ExpectedColon));
                    }
                }
                Phase::ValueStart { .. } => {
                    if character.is_whitespace() {
                        context.append(character)?;
                        consumed += length;
                    } else {
                        let cursor = ValueCursor::start(character)
                            .map_err(|_| context.syntax(ObjectSyntaxError::MissingValue))?;
                        let start = context.raw_len();
                        context.append(character)?;
                        consumed += length;
                        let Phase::ValueStart { key } =
                            std::mem::replace(&mut self.phase, Phase::AfterValue)
                        else {
                            unreachable!()
                        };
                        self.phase = Phase::Value { key, start, cursor };
                    }
                }
                Phase::Value { cursor, .. } => {
                    let (next, step) = cursor.step(character);
                    match step {
                        ValueStep::Consumed { complete } => {
                            context.append(character)?;
                            consumed += length;
                            *cursor = next;
                            if complete {
                                self.finish(context)?;
                            }
                        }
                        ValueStep::Boundary => self.finish(context)?,
                    }
                }
                Phase::AfterValue => {
                    if character.is_whitespace() {
                        context.append(character)?;
                        consumed += length;
                    } else if character == ',' {
                        context.append(character)?;
                        consumed += length;
                        self.phase = Phase::KeyOrEnd { allow_end: false };
                    } else if character == '}' {
                        context.append(character)?;
                        consumed += length;
                        self.complete = true;
                    } else {
                        return Err(context.syntax(ObjectSyntaxError::ExpectedSeparator));
                    }
                }
            }
        }
        Ok((consumed, self.complete))
    }
    /// Actual generic cursor, callbacks and fixed loop/error transports. Key
    /// payload, raw bytes and validator work are separate caller destinations.
    pub fn control_bytes<C: ObjectContext<Key = K>>() -> Option<usize> {
        let parts = [
            ValueCursor::control_bytes()?,
            size_of::<Self>(),
            size_of::<Phase<K>>(),
            size_of::<K>(),
            size_of::<C::Error>(),
            size_of::<ObjectSyntaxError>(),
            size_of::<(&mut Self, &str, &mut C)>(),
            size_of::<Result<(usize, bool), C::Error>>(),
            size_of::<Result<(), C::Error>>(),
            size_of::<Result<K, C::Error>>(),
            size_of::<Range<usize>>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<(usize, usize, char, bool)>(),
            size_of::<Option<&K>>(),
            size_of::<Option<(&K, Range<usize>)>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
