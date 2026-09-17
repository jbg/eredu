//! Mapping projections retain only a register and ordinal in the exact input.
//! Ordinary Value indexes iterable objects through its nth fallback; list
//! conversion changes the kind to sequence without copying immutable elements.
use super::*;
use crate::bounded::source::MappingProjection;

#[derive(Clone, Copy, Debug)]
pub(super) struct MappingView {
    pub register: usize,
    pub length: usize,
    pub projection: MappingProjection,
    pub listed: bool,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct MappingPair {
    pub register: usize,
    pub index: usize,
}

#[cfg(feature = "json")]
pub(super) type Pairs<'a> = crate::bounded::input::record::ReadPairs<'a>;

impl<'source, 'input, 'work> Engine<'source, 'input, 'work> {
    pub(super) fn mapping_view(
        &self,
        value: Slot,
        projection: MappingProjection,
        listed: bool,
    ) -> Result<Slot, Error> {
        if let Slot::Object(range) = value {
            return Ok(Slot::ObjectView(object::View {
                range,
                projection,
                listed,
            }));
        }
        #[cfg(feature = "json")]
        if let Slot::Structured { register, .. } = value {
            let map = self
                .borrowed
                .get(register)?
                .as_object()
                .ok_or(Error::Geometry)?;
            return Ok(Slot::Mapping(MappingView {
                register,
                length: map.len(),
                projection,
                listed,
            }));
        }
        let _ = (value, projection, listed);
        // The compact message representation does not retain arbitrary JSON
        // message key order/extra fields, so it supplies no mapping view proof.
        Err(Error::Geometry)
    }
    pub(super) fn mapping_method(
        &mut self,
        name: &str,
        arguments: Option<u16>,
    ) -> Result<(), Error> {
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let projection = match name {
            "keys" => MappingProjection::Keys,
            "values" => MappingProjection::Values,
            "items" => MappingProjection::Items,
            _ => return Err(Error::Geometry),
        };
        let value = self.pop()?;
        let view = self.mapping_view(value, projection, false)?;
        self.push(view)
    }
    pub(super) fn mapping_filter(
        &mut self,
        name: &str,
        arguments: Option<u16>,
    ) -> Result<(), Error> {
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let value = self.pop()?;
        let value = if name == "items" {
            self.mapping_view(value, MappingProjection::Items, false)?
        } else if name == "list" {
            match value {
                Slot::Object(range) => Slot::ObjectView(object::View {
                    range,
                    projection: MappingProjection::Keys,
                    listed: true,
                }),
                Slot::ObjectView(mut view) => {
                    view.listed = true;
                    Slot::ObjectView(view)
                }
                Slot::Sequence(mut value) => {
                    value.depth = 0;
                    Slot::Sequence(value)
                }
                Slot::Range(mut value) => {
                    value.listed = true;
                    Slot::Range(value)
                }
                Slot::Slice(mut value) => {
                    value.listed = true;
                    Slot::Slice(value)
                }
                Slot::Mapping(mut view) => {
                    view.listed = true;
                    Slot::Mapping(view)
                }
                Slot::Undefined | Slot::Scalar(Scalar::None) => Slot::EmptySequence,
                Slot::EmptySequence | Slot::Split(_) | Slot::MappingPair(_) | Slot::Messages => {
                    value
                }
                #[cfg(feature = "json")]
                Slot::Structured { register, .. } if self.borrowed.get(register)?.is_array() => {
                    value
                }
                Slot::Structured { .. } => {
                    self.mapping_view(value, MappingProjection::Keys, true)?
                }
                _ => return Err(Error::Geometry),
            }
        } else {
            return Err(Error::Geometry);
        };
        self.push(value)
    }
    #[cfg(feature = "json")]
    pub(super) fn mapping_pairs(&self, register: usize) -> Result<Pairs<'input>, Error> {
        Ok(Pairs::new(
            self.borrowed
                .get(register)?
                .as_object()
                .ok_or(Error::Geometry)?,
        ))
    }
    #[cfg(feature = "json")]
    pub(super) fn mapping_entry(&mut self, view: MappingView, index: usize) -> Result<Slot, Error> {
        let Some((key, value)) = self.mapping_pairs(view.register)?.nth(index) else {
            return Ok(Slot::Undefined);
        };
        Ok(match view.projection {
            MappingProjection::Keys => Slot::Text(Text::Atom(Atom::Borrowed(
                self.borrowed.set_text(key),
                Range::new(0, key.len()),
            ))),
            MappingProjection::Values => self.selected(Some(value))?,
            MappingProjection::Items => Slot::MappingPair(MappingPair {
                register: view.register,
                index,
            }),
        })
    }
    pub(super) fn mapping_item(&mut self, value: Slot, key: Slot) -> Result<Slot, Error> {
        #[cfg(feature = "json")]
        match value {
            Slot::Mapping(view) => {
                return match structured::index(key, view.length) {
                    Some(index) => self.mapping_entry(view, index),
                    None => Ok(Slot::Undefined),
                };
            }
            Slot::MappingPair(pair) => {
                let Some((name, value)) = self.mapping_pairs(pair.register)?.nth(pair.index) else {
                    return Err(Error::Geometry);
                };
                return match structured::index(key, 2) {
                    Some(0) => Ok(Slot::Text(Text::Atom(Atom::Borrowed(
                        self.borrowed.set_text(name),
                        Range::new(0, name.len()),
                    )))),
                    Some(1) => self.selected(Some(value)),
                    _ => Ok(Slot::Undefined),
                };
            }
            _ => {}
        }
        let _ = (value, key);
        Ok(Slot::Undefined)
    }
    pub(super) fn mapping_membership(
        &mut self,
        view: MappingView,
        needle: Slot,
    ) -> Result<bool, Error> {
        #[cfg(feature = "json")]
        {
            // Existing scalar equality and short-circuit iteration; pair/nested
            // container comparison remains its own ordinary comparison profile.
            if scalar(needle).is_none()
                && !matches!(needle, Slot::Undefined | Slot::Text(Text::Atom(_)))
            {
                return Err(Error::Geometry);
            }
            if view.projection == MappingProjection::Items {
                return Ok(false);
            }
            primitive::membership::sequence_contains(0..view.length, |index| {
                let selected = self.mapping_entry(view, index)?;
                if matches!(
                    selected,
                    Slot::Structured { .. } | Slot::Mapping(_) | Slot::MappingPair(_)
                ) {
                    return Ok(false);
                }
                let Slot::Bool(equal) = self.eq(needle, selected)? else {
                    return Err(Error::Geometry);
                };
                Ok(equal)
            })
        }
        #[cfg(not(feature = "json"))]
        {
            let _ = (view, needle);
            Err(Error::Geometry)
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<MappingView>(),
        size_of::<MappingPair>(),
        size_of::<MappingProjection>(),
        size_of::<(Slot, Slot, Option<u16>, usize)>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<(&str, bool, Option<usize>)>(),
    ];
    let total = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?;
    #[cfg(feature = "json")]
    let total = total
        .checked_add(crate::bounded::input::RecordValue::control_bytes()?)?
        .checked_add(size_of::<Pairs<'_>>())?
        .checked_add(size_of::<serde_json::map::Iter<'_>>())?
        .checked_add(size_of::<Option<(&str, &serde_json::Value)>>())?
        .checked_add(size_of::<(
            &serde_json::Map<String, serde_json::Value>,
            &str,
            usize,
        )>())?
        .checked_add(size_of::<Result<Pairs<'_>, Error>>())?
        .checked_add(size_of::<(
            std::ops::Range<usize>,
            &mut Engine<'_, '_, '_>,
            MappingView,
            Slot,
        )>())?;
    Some(total)
}
