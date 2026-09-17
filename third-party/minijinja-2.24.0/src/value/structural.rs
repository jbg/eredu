//! One short-circuit structural comparison driver with storage-owned continuations.
use super::{DynObject, ObjectRepr, Value};
pub(crate) enum Next<V> {
    Pair(V, V),
    Different,
    Done,
}
pub(crate) trait Storage {
    type Value;
    type Error;
    /// Compare leaves, or install the real child traversal for a container.
    fn open(&mut self, left: Self::Value, right: Self::Value) -> Result<bool, Self::Error>;
    fn next(&mut self) -> Result<Next<Self::Value>, Self::Error>;
}
pub(crate) fn equal<S: Storage>(
    storage: &mut S,
    mut left: S::Value,
    mut right: S::Value,
) -> Result<bool, S::Error> {
    loop {
        if !storage.open(left, right)? {
            return Ok(false);
        }
        match storage.next()? {
            Next::Pair(a, b) => {
                left = a;
                right = b;
            }
            Next::Different => return Ok(false),
            Next::Done => return Ok(true),
        }
    }
}
type Values = Box<dyn Iterator<Item = Value> + Send + Sync>;
type Pairs = Box<dyn Iterator<Item = (Value, Value)> + Send + Sync>;
enum Frame {
    Sequence(Values, Values),
    Map {
        pairs: Pairs,
        other: DynObject,
        count: usize,
        length_fallback: bool,
    },
}
struct Ordinary {
    frames: Vec<Frame>,
}
impl Storage for Ordinary {
    type Value = Value;
    type Error = std::convert::Infallible;
    fn open(&mut self, left: Value, right: Value) -> Result<bool, Self::Error> {
        let (Some(a), Some(b)) = (left.as_object(), right.as_object()) else {
            return Ok(left == right);
        };
        if a.is_same_object(b) {
            return Ok(true);
        }
        if a.is_same_object_type(b) {
            if let Some(order) = a.custom_cmp(b) {
                return Ok(order == std::cmp::Ordering::Equal);
            }
        }
        match (a.repr(), b.repr()) {
            (ObjectRepr::Map, ObjectRepr::Map) => {
                let length_fallback = match (a.enumerator_len(), b.enumerator_len()) {
                    (Some(a), Some(b)) => {
                        if a != b {
                            return Ok(false);
                        }
                        false
                    }
                    _ => true,
                };
                let Some(pairs) = a.try_iter_pairs() else {
                    return Ok(false);
                };
                self.frames.push(Frame::Map {
                    pairs,
                    other: b.clone(),
                    count: 0,
                    length_fallback,
                });
                Ok(true)
            }
            (ObjectRepr::Seq | ObjectRepr::Iterable, ObjectRepr::Seq | ObjectRepr::Iterable) => {
                let (Some(a), Some(b)) = (a.try_iter(), b.try_iter()) else {
                    return Ok(false);
                };
                self.frames.push(Frame::Sequence(a, b));
                Ok(true)
            }
            (ObjectRepr::Plain, ObjectRepr::Plain) => Ok(a.to_string() == b.to_string()),
            _ => Ok(false),
        }
    }
    fn next(&mut self) -> Result<Next<Value>, Self::Error> {
        loop {
            let Some(frame) = self.frames.last_mut() else {
                return Ok(Next::Done);
            };
            match frame {
                Frame::Sequence(left, right) => match (left.next(), right.next()) {
                    (Some(a), Some(b)) => return Ok(Next::Pair(a, b)),
                    (None, None) => {}
                    _ => return Ok(Next::Different),
                },
                Frame::Map {
                    pairs,
                    other,
                    count,
                    length_fallback,
                } => {
                    if let Some((key, left)) = pairs.next() {
                        *count += 1;
                        let Some(right) = other.get_value(&key) else {
                            return Ok(Next::Different);
                        };
                        return Ok(Next::Pair(left, right));
                    }
                    if *length_fallback && *count != other.try_iter().map_or(0, |iter| iter.count())
                    {
                        return Ok(Next::Different);
                    }
                }
            }
            self.frames.pop();
        }
    }
}
pub(super) fn objects_equal(left: Value, right: Value) -> bool {
    match equal(&mut Ordinary { frames: Vec::new() }, left, right) {
        Ok(value) => value,
        Err(never) => match never {},
    }
}
