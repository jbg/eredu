//! Borrowed context operands live only during the shared dispatch. The paid
//! operand/concat arrays retain indices/ranges, never input references.
use super::*;
#[cfg(feature = "json")]
use crate::bounded::input::record::{ReadKind, ReadValue};

#[cfg(feature = "json")]
#[derive(Clone, Copy, Debug)]
enum Entry<'a> {
    Json(ReadValue<'a>),
    Text(&'a str),
}
#[derive(Debug)]
pub(in crate::bounded) struct Borrowed<'a> {
    #[cfg(feature = "json")]
    values: Vec<Option<Entry<'a>>>,
    #[cfg(feature = "json")]
    result: usize,
    marker: std::marker::PhantomData<&'a ()>,
}
impl<'a> Borrowed<'a> {
    pub(in crate::bounded) fn count(geometry: super::super::source::Geometry) -> Option<usize> {
        geometry
            .operands
            .checked_add(2)?
            .checked_add(geometry.frames)?
            .checked_add(geometry.locals)
    }
    pub(in crate::bounded) fn layout(count: usize) -> Option<usize> {
        #[cfg(feature = "json")]
        {
            std::alloc::Layout::array::<Option<Entry<'a>>>(count)
                .ok()
                .map(|l| l.size())
        }
        #[cfg(not(feature = "json"))]
        {
            let _ = count;
            Some(0)
        }
    }
    pub(in crate::bounded) fn prepare(
        count: usize,
        operands: usize,
    ) -> Result<Self, std::collections::TryReserveError> {
        #[cfg(feature = "json")]
        let values = {
            let mut values = Vec::new();
            values.try_reserve_exact(count)?;
            values.resize(count, None);
            values
        };
        #[cfg(not(feature = "json"))]
        let _ = (count, operands);
        Ok(Self {
            #[cfg(feature = "json")]
            values,
            #[cfg(feature = "json")]
            result: operands,
            marker: std::marker::PhantomData,
        })
    }
    pub(in crate::bounded) fn clear(&mut self) {
        #[cfg(feature = "json")]
        self.values.fill(None);
    }

    pub(super) fn text(&self, index: usize) -> Option<&'a str> {
        #[cfg(feature = "json")]
        {
            match self.values.get(index).copied().flatten()? {
                Entry::Json(value) => value.as_str(),
                Entry::Text(value) => Some(value),
            }
        }
        #[cfg(not(feature = "json"))]
        {
            let _ = index;
            None
        }
    }
    pub(super) fn push(&mut self, value: Slot, depth: usize) -> Result<Slot, Error> {
        if let Slot::Slice(mut view) = value {
            view.base = slice::Base::from_slot(self.push(view.base.slot(), depth)?)?;
            return Ok(Slot::Slice(view));
        }
        let origin = match value {
            Slot::Structured { register, .. }
            | Slot::Mapping(MappingView { register, .. })
            | Slot::MappingPair(MappingPair { register, .. })
            | Slot::Text(Text::Atom(Atom::Borrowed(register, _)))
            | Slot::Split(SplitView {
                value: Atom::Borrowed(register, _),
                ..
            }) => Some(register),
            _ => None,
        };
        #[cfg(feature = "json")]
        {
            let entry = match origin {
                Some(index) => Some(
                    self.values
                        .get(index)
                        .copied()
                        .flatten()
                        .ok_or(Error::Geometry)?,
                ),
                None => None,
            };
            *self.values.get_mut(depth).ok_or(Error::Geometry)? = entry;
        }
        #[cfg(not(feature = "json"))]
        if origin.is_some() {
            return Err(Error::Geometry);
        }
        Ok(match value {
            Slot::Mapping(mut view) => {
                view.register = depth;
                Slot::Mapping(view)
            }
            Slot::MappingPair(mut pair) => {
                pair.register = depth;
                Slot::MappingPair(pair)
            }

            Slot::Structured { length, .. } => Slot::Structured {
                length,
                register: depth,
            },
            Slot::Text(Text::Atom(Atom::Borrowed(_, range))) => {
                Slot::Text(Text::Atom(Atom::Borrowed(depth, range)))
            }
            Slot::Split(SplitView {
                value: Atom::Borrowed(_, range),
                separator,
                length,
                limit,
            }) => Slot::Split(SplitView {
                value: Atom::Borrowed(depth, range),
                separator,
                length,
                limit,
            }),
            other => other,
        })
    }
    #[cfg(feature = "json")]
    fn set(&mut self, value: ReadValue<'a>) -> usize {
        self.values[self.result] = Some(Entry::Json(value));
        self.result
    }
    #[cfg(feature = "json")]
    pub(super) fn set_text(&mut self, value: &'a str) -> usize {
        self.values[self.result] = Some(Entry::Text(value));
        self.result
    }
    #[cfg(feature = "json")]
    pub(super) fn get(&self, index: usize) -> Result<ReadValue<'a>, Error> {
        match self.values.get(index).copied().flatten() {
            Some(Entry::Json(value)) => Ok(value),
            _ => Err(Error::Geometry),
        }
    }
}
pub(super) fn index(key: Slot, len: usize) -> Option<usize> {
    primitive::sequence_index(
        scalar(key)
            .and_then(primitive::scalar::signed)
            .and_then(|v| i64::try_from(v).ok()),
        || Some(len),
    )
}
impl<'source, 'input, 'work> Engine<'source, 'input, 'work> {
    #[cfg(feature = "json")]
    pub(super) fn structured(&mut self, length: usize, value: ReadValue<'input>) -> Slot {
        Slot::Structured {
            length,
            register: self.borrowed.set(value),
        }
    }
    #[cfg(feature = "json")]
    pub(super) fn selected(&mut self, value: Option<ReadValue<'input>>) -> Result<Slot, Error> {
        Ok(match value {
            None => Slot::Undefined,
            Some(value) => match value.kind() {
                ReadKind::Null => Slot::Scalar(Scalar::None),
                ReadKind::Bool(v) => Slot::Scalar(Scalar::Bool(v)),
                ReadKind::Number(v) => Slot::Scalar(if let Some(v) = v.as_u64() {
                    Scalar::U64(v)
                } else if let Some(v) = v.as_i64() {
                    Scalar::I64(v)
                } else if let Some(v) = v.as_f64() {
                    Scalar::F64(v)
                } else {
                    return Err(Error::Geometry);
                }),
                ReadKind::Text(text) => Slot::Text(Text::Atom(Atom::Borrowed(
                    self.borrowed.set(value),
                    Range::new(0, text.len()),
                ))),
                ReadKind::Array(items) => self.structured(items.len(), value),
                ReadKind::Object(items) => self.structured(items.len(), value),
            },
        })
    }

    pub(super) fn structured_attr(&mut self, value: Slot, name: &str) -> Result<Slot, Error> {
        if let Slot::Object(range) = value {
            return self.object_attr(range, name);
        }
        #[cfg(feature = "json")]
        if let Slot::Structured { register, .. } = value {
            let source = self.borrowed.get(register)?;
            return self.selected(source.as_object().and_then(|map| map.get(name)));
        }
        // Lenient ordinary behavior: a missing item is undefined; looking up
        // another member on an already undefined base is an error.
        if matches!(value, Slot::Undefined) {
            Err(Error::Geometry)
        } else {
            Ok(Slot::Undefined)
        }
    }
    pub(super) fn selected_item(&mut self, value: Slot, key: Slot) -> Result<Slot, Error> {
        if let Slot::Object(range) = value {
            return self.object_get(range, key);
        }
        if let Slot::ObjectView(view) = value {
            return index(key, view.range.len).map_or(Ok(Slot::Undefined), |index| {
                self.object_view_item(view, index)
            });
        }
        if let Slot::Sequence(range) = value {
            return index(key, range.len).map_or(Ok(Slot::Undefined), |index| {
                self.sequence_item(range, index)
            });
        }
        if let Slot::Slice(view) = value {
            return self.slice_item(view, key);
        }
        if let Slot::Range(value) = value {
            // Ordinary get_item_opt indexes Iterable through its exact
            // enumerator length/nth fallback as well as materialized lists.
            return index(key, value.plan.length)
                .map_or(Ok(Slot::Undefined), |index| value.item(index));
        }
        if matches!(value, Slot::Mapping(_) | Slot::MappingPair(_)) {
            return self.mapping_item(value, key);
        }
        if let Slot::Split(view) = value {
            return self.split_item(view, index(key, view.length));
        }
        if let Slot::Messages = value {
            let Some(index) = index(key, self.context.messages().len())
                .filter(|&i| i < self.context.messages().len())
            else {
                return Ok(Slot::Undefined);
            };
            #[cfg(feature = "json")]
            if let Some(value) = self.context.messages().value(index) {
                return self.selected(Some(value));
            }
            return Ok(Slot::Message(index));
        }
        #[cfg(feature = "json")]
        if let Slot::Structured { register, .. } = value {
            let source = self.borrowed.get(register)?;
            let selected = match source.kind() {
                ReadKind::Array(values) => index(key, values.len()).and_then(|i| values.get(i)),
                ReadKind::Object(values) => match key {
                    Slot::Text(Text::Atom(atom)) => values.get(atom_text(
                        self.source,
                        self.context,
                        &self.borrowed,
                        self.workspace.context_text.as_deref(),
                        atom,
                    )?),
                    Slot::Text(Text::Concat { .. }) => return Err(Error::Geometry),
                    _ => None,
                },
                _ => return Err(Error::Geometry),
            };
            return self.selected(selected);
        }
        if matches!(value, Slot::Undefined | Slot::Loop(_)) {
            return Err(Error::Geometry);
        }
        // A non-container has no selected value. String indexing is a separate
        // ordinary operation; do not silently replace its nonempty result.
        if matches!(value, Slot::Text(_)) {
            return Err(Error::Geometry);
        }
        Ok(Slot::Undefined)
    }
    /// Snapshot a borrowed atom only when concatenation must retain it. The
    /// same append measures every copied byte before this seventh destination.
    pub(super) fn own_atom(&mut self, atom: Atom) -> Result<Atom, Error> {
        if !matches!(atom, Atom::Borrowed(..)) {
            return Ok(atom);
        }
        let text = atom_text(self.source, self.context, &self.borrowed, None, atom)?;
        let start = self.counts.context_text;
        let end = start.checked_add(text.len()).ok_or(Error::Overflow)?;
        if let Some(bytes) = self.workspace.context_text.as_deref_mut() {
            bytes
                .get_mut(start..end)
                .ok_or(Error::TextCapacity(super::super::render::TextCapacity {
                    atoms: self.counts.concat,
                    bytes: end,
                }))?
                .copy_from_slice(text.as_bytes());
        }
        self.counts.context_text = end;
        Ok(Atom::Owned(Range::new(start, text.len())))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let parts = [
        size_of::<Borrowed<'_>>(),
        size_of::<Result<Borrowed<'_>, std::collections::TryReserveError>>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<(usize, usize)>(),
        #[cfg(feature = "json")]
        size_of::<Vec<Option<Entry<'_>>>>(),
        size_of::<TypeFacts>(),
        size_of::<primitive::type_tests::TypeTest>(),
        size_of::<Option<primitive::type_tests::TypeTest>>(),
        size_of::<(&str, &TypeFacts)>(),
        size_of::<(&Engine<'_, '_, '_>, primitive::type_tests::TypeTest, Slot)>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Option<Slot>>(),
        size_of::<(Slot, Slot, usize)>(),
        size_of::<(Option<i64>, Option<usize>, isize)>(),
        size_of::<(Atom, &str, usize, usize)>(),
        size_of::<Result<Slot, Error>>(),
    ];
    let total = parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
    #[cfg(feature = "json")]
    let total = total
        .checked_add(crate::bounded::input::RecordValue::control_bytes()?)?
        .checked_add(size_of::<Option<ReadValue<'_>>>())?
        .checked_add(size_of::<(ReadValue<'_>, &str, Option<ReadValue<'_>>)>())?
        .checked_add(size_of::<Option<&serde_json::Value>>())?
        .checked_add(size_of::<(
            &serde_json::Value,
            &str,
            Option<&serde_json::Value>,
        )>())?;
    Some(total)
}

// Fixed facts of the actual closed representation, without dynamic Value or
// iterator allocation. Every represented JSON/message/loop container exposes
// its ordinary enumerable members; None/undefined have the ordinary empty iter.
struct TypeFacts {
    kind: crate::value::ValueKind,
    integer: bool,
    float: bool,
    boolean: Option<bool>,
}
impl primitive::type_tests::Facts for TypeFacts {
    fn kind(&self) -> crate::value::ValueKind {
        self.kind
    }
    fn integer(&self) -> bool {
        self.integer
    }
    fn float(&self) -> bool {
        self.float
    }
    fn boolean(&self) -> Option<bool> {
        self.boolean
    }
    fn iterable(&self) -> bool {
        use crate::value::ValueKind as K;
        matches!(
            self.kind,
            K::Undefined | K::None | K::String | K::Seq | K::Map | K::Iterable
        )
    }
}
impl Engine<'_, '_, '_> {
    pub(super) fn type_test(
        &self,
        test: primitive::type_tests::TypeTest,
        value: Slot,
    ) -> Result<bool, Error> {
        use crate::value::ValueKind as K;
        let facts = if let Some(value) = scalar(value) {
            match value {
                Scalar::None => TypeFacts {
                    kind: K::None,
                    integer: false,
                    float: false,
                    boolean: None,
                },
                Scalar::Bool(value) => TypeFacts {
                    kind: K::Bool,
                    integer: false,
                    float: false,
                    boolean: Some(value),
                },
                Scalar::F64(_) => TypeFacts {
                    kind: K::Number,
                    integer: false,
                    float: true,
                    boolean: None,
                },
                Scalar::U64(_) | Scalar::I64(_) | Scalar::U128(_) | Scalar::I128(_) => TypeFacts {
                    kind: K::Number,
                    integer: true,
                    float: false,
                    boolean: None,
                },
            }
        } else {
            let kind = match value {
                Slot::Undefined => K::Undefined,
                #[cfg(feature = "chat-clock")]
                Slot::Clock => K::Plain,
                Slot::Namespace(_) | Slot::Kwargs(_) | Slot::Macro(_) | Slot::Object(_) => K::Map,
                Slot::ObjectView(view) => {
                    if view.listed {
                        K::Seq
                    } else {
                        K::Iterable
                    }
                }
                Slot::Closure(_) | Slot::Arguments { .. } => return Err(Error::Geometry),
                Slot::Text(_) => K::String,
                Slot::Messages | Slot::EmptySequence | Slot::Split(_) | Slot::MappingPair(_) => {
                    K::Seq
                }
                Slot::Sequence(value) => {
                    if value.depth == 0 {
                        K::Seq
                    } else {
                        K::Iterable
                    }
                }
                Slot::Range(value) => {
                    if value.listed {
                        K::Seq
                    } else {
                        K::Iterable
                    }
                }
                Slot::Slice(value) => {
                    if value.listed {
                        K::Seq
                    } else {
                        K::Iterable
                    }
                }
                Slot::Mapping(view) => {
                    if view.listed {
                        K::Seq
                    } else {
                        K::Iterable
                    }
                }
                Slot::Message(_) | Slot::Loop(_) => K::Map,
                #[cfg(feature = "json")]
                Slot::Structured { register, .. } => match self.borrowed.get(register)?.kind() {
                    ReadKind::Array(_) => K::Seq,
                    ReadKind::Object(_) => K::Map,
                    _ => return Err(Error::Geometry),
                },
                _ => return Err(Error::Geometry),
            };
            TypeFacts {
                kind,
                integer: false,
                float: false,
                boolean: None,
            }
        };
        Ok(test.check(&facts))
    }
}
