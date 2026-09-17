use crate::cmp;
use ahash::{AHashSet, AHasher};
use serde_json::Value;
use std::{
    borrow::Borrow,
    hash::{Hash, Hasher},
};

// Based on implementation proposed by Sven Marnach:
// https://stackoverflow.com/questions/60882381/what-is-the-fastest-correct-way-to-detect-that-there-are-no-duplicates-in-a-json
pub(crate) struct HashedValue<'a>(&'a Value);

impl PartialEq for HashedValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        cmp::equal(self.0, other.0)
    }
}

impl Eq for HashedValue<'_> {}

impl Hash for HashedValue<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let complete = hash_node_with::<crate::SerdeJson, _, _, _>(
            &self.0, state, &mut AHasher::default, &mut |_| true);
        debug_assert!(complete);
    }
}

// Empirically calculated threshold after which the validator resorts to hashing.
// Calculated for an array of mixed types, large homogeneous arrays of primitive values might be
// processed faster with different thresholds, but this one gives a good baseline for the common
// case.
pub(crate) const ITEMS_SIZE_THRESHOLD: usize = 15;

// Generic over `Borrow<Value>` so both borrowed `serde_json` slices and the `Cow<Value>` handles
// materialized from other representations run the same duplicate-detection algorithm.
#[inline]
#[must_use]
pub fn is_unique<T: Borrow<Value>>(items: &[T]) -> bool {
    let mut seen = None;
    is_unique_with(items.len(), &mut |step| match step {
        UniqueStep::Controls(_) => true,
        UniqueStep::Compare(a, b) => !cmp::equal(items[a].borrow(), items[b].borrow()),
        UniqueStep::HashTable(size) => { seen = Some(AHashSet::with_capacity(size)); true }
        UniqueStep::Insert(index) => seen.as_mut().expect("hash table step").insert(HashedValue(items[index].borrow())),
    })
}

/// One reached step of the ordinary duplicate-detection algorithm.
#[derive(Clone, Copy, Debug)]
pub enum UniqueStep {
    /// Fixed shared driver controls; None indicates overflow.
    Controls(Option<usize>),
    /// Return true exactly when these two elements differ.
    Compare(usize, usize),
    /// Prepare a table for this reached input length, before hashing begins.
    HashTable(usize),
    /// Return true exactly when inserting this element found no duplicate.
    Insert(usize),
}

/// Same small-array shortcuts, comparison order and hashing threshold as the
/// ordinary slice worker. False from any callback terminates immediately; a
/// funded caller retains the separate first refusal before interpreting false.
pub fn is_unique_with<P: FnMut(UniqueStep) -> bool>(size: usize, step: &mut P) -> bool {
    use std::mem::{size_of, size_of_val};
    let parts = [size_of::<(usize, usize, usize, bool)>(), size_of::<UniqueStep>(),
        size_of::<(&mut P, usize)>(), size_of::<Option<usize>>()];
    if !step(UniqueStep::Controls(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add))) { return false; }
    if size <= 1 { true }
    else if size == 2 { step(UniqueStep::Compare(0, 1)) }
    else if size == 3 {
        step(UniqueStep::Compare(0, 1)) && step(UniqueStep::Compare(0, 2)) && step(UniqueStep::Compare(1, 2))
    } else if size <= ITEMS_SIZE_THRESHOLD {
        let mut idx = 0;
        while idx < size {
            let mut inner_idx = idx + 1;
            while inner_idx < size {
                if !step(UniqueStep::Compare(idx, inner_idx)) { return false; }
                inner_idx += 1;
            }
            idx += 1;
        }
        true
    } else {
        if !step(UniqueStep::HashTable(size)) { return false; }
        (0..size).all(|index| step(UniqueStep::Insert(index)))
    }
}

/// Borrowed form of the ordinary value hashing worker. Primitive encoding,
/// array order and object member XOR are unchanged. The original caller lends
/// a cold initialized object hasher and pays before every recursive frame.
pub fn hash_node_with<F, H, B, P>(node: &F::Node<'_>, state: &mut H, build: &mut B, permit: &mut P) -> bool
where F: crate::Json, H: Hasher, B: FnMut() -> AHasher, P: FnMut(Option<usize>) -> bool,
{
    use crate::{Node, Object, Array, JsonNumber, JsonType};
    use std::mem::{size_of, size_of_val};
    let parts = [size_of::<(&F::Node<'_>, &mut H, &mut B, &mut P)>(),
        size_of::<JsonType>(), size_of::<Option<bool>>(), size_of::<Option<f64>>(),
        size_of::<Option<u64>>(), size_of::<Option<i64>>(), size_of::<Option<std::borrow::Cow<'_,str>>>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Number>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Array>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Object>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Array as Array<'_,F>>::ElementsIter>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Object as Object<'_,F>>::MembersIter>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Object as Object<'_,F>>::MemberName>(),
        size_of::<F::Node<'_>>(), size_of::<AHasher>(), size_of::<(u64,usize,bool)>(),
        size_of::<(&str,&mut H)>(), size_of::<(&str,&mut AHasher)>()];
    if !permit(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)) { return false; }
    match node.json_type() {
        JsonType::Null => state.write_u32(3_221_225_473),
        JsonType::Boolean => node.as_boolean().expect("boolean node").hash(state),
        JsonType::Number | JsonType::Integer => {
            let item = node.as_number().expect("number node");
            if let Some(value) = item.as_f64() { value.to_bits().hash(state); }
            else if let Some(value) = item.as_u64() { value.hash(state); }
            else if let Some(value) = item.as_i64() { value.hash(state); }
        }
        JsonType::String => node.as_string().expect("string node").as_ref().hash(state),
        JsonType::Array => for child in node.as_array().expect("array node").elements() {
            if !hash_node_with::<F, H, B, P>(&child, state, build, permit) { return false; }
        },
        JsonType::Object => {
            let mut hash = 0;
            for (key, child) in node.as_object().expect("object node").members() {
                let mut item_hasher = build();
                key.as_ref().hash(&mut item_hasher);
                if !hash_node_with::<F, AHasher, B, P>(&child, &mut item_hasher, build, permit) { return false; }
                hash ^= item_hasher.finish();
            }
            state.write_u64(hash);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{is_unique, ITEMS_SIZE_THRESHOLD};
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&[] => true; "empty array")]
    #[test_case(&[json!(1)] => true; "one element array")]
    #[test_case(&[json!(1), json!(2)] => true; "two unique elements")]
    #[test_case(&[json!(1), json!(1)] => false; "two non-unique elements")]
    #[test_case(&[json!(1), json!(2), json!(3)] => true; "three unique elements")]
    #[test_case(&[json!(1), json!(2), json!(1)] => false; "three non-unique elements")]
    #[test_case(&[json!(1), json!(2), json!(3), json!(4), json!(5), json!(6), json!(7), json!(8), json!(9), json!(10), json!(11), json!(12), json!(13), json!(14), json!(15), json!(1.0)] => false; "positive numbers with fractions")]
    #[test_case(&[json!(-1), json!(-2), json!(-3), json!(-4), json!(-5), json!(-6), json!(-7), json!(-8), json!(-9), json!(-10), json!(-11), json!(-12), json!(-13), json!(-14), json!(-15), json!(-1.0)] => false; "negative numbers with fractions")]
    #[test_case(&[json!(1), json!("string"), json!(true), json!(null), json!({"key": "value"}), json!([1, 2, 3])] => true; "mixed types")]
    #[test_case(&[json!({"a": 1, "b": 1}), json!({"a": 1, "b": 2}), json!({"a": 1, "b": 3})] => true; "complex objects unique")]
    #[test_case(&[json!({"a": 1, "b": 2}), json!({"b": 2, "a": 1}), json!({"a": 1, "b": 2})] => false; "complex objects non-unique")]
    fn test_is_unique(items: &[Value]) -> bool {
        is_unique(items)
    }

    #[test_case(ITEMS_SIZE_THRESHOLD => true; "small array unique")]
    #[test_case(ITEMS_SIZE_THRESHOLD + 1 => true; "large array unique")]
    fn test_unique_arrays(size: usize) -> bool {
        let arr = (1..=size).map(|i| json!(i)).collect::<Vec<_>>();
        is_unique(&arr)
    }

    #[test_case(ITEMS_SIZE_THRESHOLD => false; "small array non-unique")]
    #[test_case(ITEMS_SIZE_THRESHOLD + 1 => false; "large array non-unique")]
    fn test_non_unique_arrays(size: usize) -> bool {
        let mut arr = (1..=size).map(|i| json!(i)).collect::<Vec<_>>();
        arr[size - 1] = json!(1);
        is_unique(&arr)
    }
}
