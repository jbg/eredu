use super::*;
use std::alloc::Layout;

pub use crate::bounded::render::JsonCapacity as Capacity;
impl Capacity {
    /// Pure exact Rust destination layouts, with no input walk or allocation.
    pub fn bytes(self) -> Option<usize> {
        self.bytes_for::<()>()
    }
    pub(crate) fn bytes_for<V: Copy>(self) -> Option<usize> {
        Layout::array::<Frame<'_, V>>(self.frames)
            .ok()?
            .size()
            .checked_add(
                Layout::array::<Option<Entry<'_, V>>>(self.keys)
                    .ok()?
                    .size(),
            )
    }
}
#[derive(Clone, Copy, Debug)]
struct Entry<'a, V> {
    key: Key<'a, V>,
    value: Node<'a, V>,
}
#[derive(Debug)]
enum Container<'a, V> {
    Empty,
    Record {
        value: crate::bounded::input::RecordValue<'a>,
        next: usize,
        end: usize,
    },
    Custom {
        value: V,
        next: usize,
        end: usize,
    },
    Array(std::slice::Iter<'a, Value>),
    Object(serde_json::map::Iter<'a>),
    Sorted {
        next: usize,
        end: usize,
    },
}
#[derive(Debug)]
struct Frame<'a, V> {
    container: Container<'a, V>,
    key_floor: usize,
    first: bool,
    delimiter: char,
}
impl<V> Default for Frame<'_, V> {
    fn default() -> Self {
        Self {
            container: Container::Empty,
            key_floor: 0,
            first: true,
            delimiter: ']',
        }
    }
}
pub(super) enum Step<'a, V> {
    Done,
    Entry {
        key: Option<Key<'a, V>>,
        value: Node<'a, V>,
        first: bool,
        depth: usize,
    },
    Close {
        delimiter: char,
        depth: usize,
    },
}

/// Temporary traversal owners borrow the input only until the writer returns.
/// They never enter the owned rendered-string or error payload.
#[derive(Debug)]
pub struct Scratch<'a, V = ()> {
    frames: Vec<Frame<'a, V>>,
    keys: Vec<Option<Entry<'a, V>>>,
    used: Capacity,
    grow: bool,
}
impl<'a, V: Copy> Scratch<'a, V> {
    /// Ordinary allocation policy for the same streaming formatter.
    pub fn ordinary() -> Self {
        Self {
            frames: Vec::new(),
            keys: Vec::new(),
            used: Capacity::default(),
            grow: true,
        }
    }
    /// Allocate only the supplied already admitted fixed destinations.
    pub fn fixed(capacity: Capacity) -> Result<Self, Error> {
        capacity.bytes_for::<V>().ok_or(Error::Overflow)?;
        let mut value = Self::ordinary();
        value.ensure(capacity)?;
        value.grow = false;
        Ok(value)
    }
    /// Actual logical destination capacities used for a checked attempt.
    pub fn capacity(&self) -> Capacity {
        Capacity {
            frames: self.frames.len(),
            keys: self.keys.len(),
        }
    }
    fn ensure(&mut self, need: Capacity) -> Result<(), Error> {
        let capacity = self.capacity();
        if need.frames <= capacity.frames && need.keys <= capacity.keys {
            return Ok(());
        }
        let need = Capacity {
            frames: need.frames.max(capacity.frames),
            keys: need.keys.max(capacity.keys),
        };
        need.bytes_for::<V>().ok_or(Error::Overflow)?;
        if !self.grow {
            return Err(Error::Capacity(need));
        }
        if need.frames > self.frames.len() {
            self.frames
                .try_reserve_exact(need.frames - self.frames.len())
                .map_err(Error::Reserve)?;
            self.frames.resize_with(need.frames, Frame::default);
        }
        if need.keys > self.keys.len() {
            self.keys
                .try_reserve_exact(need.keys - self.keys.len())
                .map_err(Error::Reserve)?;
            self.keys.resize_with(need.keys, || None);
        }
        Ok(())
    }
    pub(super) fn clear(&mut self) {
        for frame in &mut self.frames[..self.used.frames] {
            *frame = Frame::default();
        }
        for entry in &mut self.keys[..self.used.keys] {
            *entry = None;
        }
        self.used = Capacity::default();
    }
    pub(super) fn push_array(&mut self, values: &'a [Value]) -> Result<(), Error> {
        let need = Capacity {
            frames: self.used.frames.checked_add(1).ok_or(Error::Overflow)?,
            keys: self.used.keys,
        };
        self.ensure(need)?;
        self.frames[self.used.frames] = Frame {
            container: Container::Array(values.iter()),
            key_floor: self.used.keys,
            first: true,
            delimiter: ']',
        };
        self.used = need;
        Ok(())
    }
    pub(super) fn push(&mut self, value: &'a Value, sort: bool) -> Result<(), Error> {
        let sorted = match value {
            // Already ordered maps (including serde's default BTreeMap) never
            // allocate or rescan sorted-key storage.
            Value::Object(values) if sort => !values
                .keys()
                .zip(values.keys().skip(1))
                .all(|(a, b)| a <= b),
            _ => false,
        };
        let count = if sorted {
            value.as_object().expect("sorted object").len()
        } else {
            0
        };
        let need = Capacity {
            frames: self.used.frames.checked_add(1).ok_or(Error::Overflow)?,
            keys: self.used.keys.checked_add(count).ok_or(Error::Overflow)?,
        };
        self.ensure(need)?;
        let key_floor = self.used.keys;
        let (container, delimiter) = match value {
            Value::Array(values) => (Container::Array(values.iter()), ']'),
            Value::Object(values) if sorted => {
                for (slot, (key, value)) in self.keys[key_floor..need.keys]
                    .iter_mut()
                    .zip(values.iter())
                {
                    *slot = Some(Entry {
                        key: Key::Text(key.as_str()),
                        value: Node::Json(value),
                    });
                }
                sort_entries(&mut self.keys[key_floor..need.keys], |a, b| match (a, b) {
                    (Key::Text(a), Key::Text(b)) => Ok(a.cmp(b)),
                    _ => Err(Error::Write(fmt::Error)),
                })?;
                (
                    Container::Sorted {
                        next: key_floor,
                        end: need.keys,
                    },
                    '}',
                )
            }
            Value::Object(values) => (Container::Object(values.iter()), '}'),
            _ => return Err(Error::Overflow),
        };
        self.frames[self.used.frames] = Frame {
            container,
            key_floor,
            first: true,
            delimiter,
        };
        self.used = need;
        Ok(())
    }

    pub(super) fn push_record(
        &mut self,
        value: crate::bounded::input::RecordValue<'a>,
        sort: bool,
    ) -> Result<(), Error> {
        use crate::bounded::input::RecordValue as R;
        // Borrowed declarations may share nested records. Bound traversal by
        // the existing VM recursion policy even for a cyclic static declaration.
        if self.used.frames >= crate::environment::MAX_RECURSION {
            return Err(Error::Overflow);
        }
        let length = value.length().ok_or(Error::Overflow)?;
        let object = value.is_object();
        let sorted = object && sort;
        let need = Capacity {
            frames: self.used.frames.checked_add(1).ok_or(Error::Overflow)?,
            keys: self
                .used
                .keys
                .checked_add(if sorted { length } else { 0 })
                .ok_or(Error::Overflow)?,
        };
        self.ensure(need)?;
        let key_floor = self.used.keys;
        let container = if sorted {
            for (slot, (key, value)) in self.keys[key_floor..need.keys]
                .iter_mut()
                .zip(value.fields().ok_or(Error::Overflow)?)
            {
                *slot = Some(Entry {
                    key: Key::Text(key),
                    value: Node::Record(value),
                });
            }
            sort_entries(&mut self.keys[key_floor..need.keys], |a, b| match (a, b) {
                (Key::Text(a), Key::Text(b)) => Ok(a.cmp(b)),
                _ => Err(Error::Write(fmt::Error)),
            })?;
            Container::Sorted {
                next: key_floor,
                end: need.keys,
            }
        } else {
            if !matches!(value, R::Array(_) | R::Object(_)) {
                return Err(Error::Overflow);
            }
            Container::Record {
                value,
                next: 0,
                end: length,
            }
        };
        self.frames[self.used.frames] = Frame {
            container,
            key_floor,
            first: true,
            delimiter: if object { '}' } else { ']' },
        };
        self.used = need;
        Ok(())
    }

    pub(super) fn push_view(
        &mut self,
        value: V,
        object: bool,
        length: usize,
        sort: bool,
        view: &impl View<'a, V>,
    ) -> Result<(), Error> {
        let sorted = object && sort;
        let need = Capacity {
            frames: self.used.frames.checked_add(1).ok_or(Error::Overflow)?,
            keys: self
                .used
                .keys
                .checked_add(if sorted { length } else { 0 })
                .ok_or(Error::Overflow)?,
        };
        self.ensure(need)?;
        let key_floor = self.used.keys;
        let container = if sorted {
            for index in 0..length {
                let (key, value) = view.entry(value, index)?;
                self.keys[key_floor + index] = Some(Entry {
                    key: key.ok_or(Error::Write(fmt::Error))?,
                    value,
                });
            }
            sort_entries(&mut self.keys[key_floor..need.keys], |a, b| {
                Ok(a.text(view)?.cmp(b.text(view)?))
            })?;
            Container::Sorted {
                next: key_floor,
                end: need.keys,
            }
        } else {
            Container::Custom {
                value,
                next: 0,
                end: length,
            }
        };
        self.frames[self.used.frames] = Frame {
            container,
            key_floor,
            first: true,
            delimiter: if object { '}' } else { ']' },
        };
        self.used = need;
        Ok(())
    }
    pub(super) fn step(&mut self, view: &impl View<'a, V>) -> Result<Step<'a, V>, Error> {
        let Some(index) = self.used.frames.checked_sub(1) else {
            return Ok(Step::Done);
        };
        let frame = &mut self.frames[index];
        let entry = match &mut frame.container {
            Container::Record { value, next, end } if *next < *end => {
                let entry = if value.is_object() {
                    value
                        .fields()
                        .and_then(|mut fields| fields.nth(*next))
                        .map(|(key, value)| (Some(Key::Text(key)), Node::Record(value)))
                } else {
                    value.item(*next).map(|value| (None, Node::Record(value)))
                };
                *next += 1;
                entry
            }
            Container::Record { .. } => None,
            Container::Array(values) => values.next().map(|value| (None, Node::Json(value))),
            Container::Object(values) => values
                .next()
                .map(|(key, value)| (Some(Key::Text(key.as_str())), Node::Json(value))),
            Container::Sorted { next, end } if *next < *end => {
                let entry = self.keys[*next].as_ref().ok_or(Error::Overflow)?;
                *next += 1;
                Some((Some(entry.key), entry.value))
            }
            Container::Custom { value, next, end } if *next < *end => {
                let entry = view.entry(*value, *next)?;
                *next += 1;
                Some(entry)
            }
            Container::Sorted { .. } | Container::Custom { .. } | Container::Empty => None,
        };
        if let Some((key, value)) = entry {
            let first = std::mem::replace(&mut frame.first, false);
            return Ok(Step::Entry {
                key,
                value,
                first,
                depth: self.used.frames,
            });
        }
        let delimiter = frame.delimiter;
        let key_floor = frame.key_floor;
        *frame = Frame::default();
        for entry in &mut self.keys[key_floor..self.used.keys] {
            *entry = None;
        }
        self.used = Capacity {
            frames: index,
            keys: key_floor,
        };
        Ok(Step::Close {
            delimiter,
            depth: index,
        })
    }
}
// Heap selection keeps one fixed index tuple instead of a recursive library
// sorting stack. JSON object keys are unique, so stable/unstable tie policy
// cannot alter the registered formatter's lexical order.

fn sort_entries<'a, V: Copy, F>(values: &mut [Option<Entry<'a, V>>], key: F) -> Result<(), Error>
where
    F: Fn(&Key<'a, V>, &Key<'a, V>) -> Result<std::cmp::Ordering, Error>,
{
    fn sift<'a, V: Copy, F>(
        values: &mut [Option<Entry<'a, V>>],
        mut root: usize,
        end: usize,
        key: &F,
    ) -> Result<(), Error>
    where
        F: Fn(&Key<'a, V>, &Key<'a, V>) -> Result<std::cmp::Ordering, Error>,
    {
        loop {
            let Some(mut child) = root
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .filter(|&n| n < end)
            else {
                return Ok(());
            };
            if child + 1 < end
                && key(
                    &values[child].as_ref().ok_or(Error::Overflow)?.key,
                    &values[child + 1].as_ref().ok_or(Error::Overflow)?.key,
                )?
                .is_lt()
            {
                child += 1;
            }
            if !key(
                &values[root].as_ref().ok_or(Error::Overflow)?.key,
                &values[child].as_ref().ok_or(Error::Overflow)?.key,
            )?
            .is_lt()
            {
                return Ok(());
            }
            values.swap(root, child);
            root = child;
        }
    }
    for root in (0..values.len() / 2).rev() {
        sift(values, root, values.len(), &key)?;
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift(values, 0, end, &key)?;
    }
    Ok(())
}

pub(super) fn control_bytes<V: Copy>() -> Option<usize> {
    let parts = [
        size_of::<Scratch<'_, V>>(),
        size_of::<Capacity>(),
        size_of::<Key<'_, V>>(),
        size_of::<Node<'_, V>>(),
        size_of::<(V, bool, usize, bool)>(),
        size_of::<Result<Step<'_, V>, Error>>(),
        size_of::<Frame<'_, V>>(),
        size_of::<Entry<'_, V>>(),
        size_of::<Container<'_, V>>(),
        size_of::<Step<'_, V>>(),
        size_of::<Option<Entry<'_, V>>>(),
        size_of::<Option<usize>>(),
        size_of::<(&Value, bool, usize, usize)>(),
        size_of::<Result<Scratch<'_, V>, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Layout>(),
        size_of::<std::slice::IterMut<'_, Option<Entry<'_, V>>>>(),
        size_of::<std::slice::IterMut<'_, Frame<'_, V>>>(),
        size_of::<
            std::iter::Zip<serde_json::map::Keys<'_>, std::iter::Skip<serde_json::map::Keys<'_>>>,
        >(),
        size_of::<
            std::iter::Zip<
                std::slice::IterMut<'_, Option<Entry<'_, V>>>,
                serde_json::map::Iter<'_>,
            >,
        >(),
        size_of::<(&mut [Option<Entry<'_, V>>], usize, usize, usize)>(),
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        size_of::<Option<usize>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
