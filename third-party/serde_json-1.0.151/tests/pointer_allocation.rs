use serde_json::{
    allocation::{Allocation, AllocationError, Unenforced},
    json, Value,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Refuse {
    calls: AtomicUsize,
    at: usize,
}
impl Allocation for Refuse {
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == self.at {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}

#[test]
fn paid_pointer_preserves_two_pass_escape_semantics() {
    let alphabet = ['~', '0', '1', '/', '水'];
    for length in 0..=5 {
        for mut word in 0..alphabet.len().pow(length) {
            let mut component = String::new();
            for _ in 0..length {
                component.push(alphabet[word % alphabet.len()]);
                word /= alphabet.len();
            }
            let expected = component.replace("~1", "/").replace("~0", "~");
            if component.contains('/') {
                continue;
            }
            let mut map = serde_json::Map::new();
            map.insert(expected, json!(7));
            let source = Value::Object(map);
            let pointer = format!("/{component}");
            assert_eq!(
                source
                    .pointer_with_allocations(&pointer, &Unenforced)
                    .unwrap(),
                Some(&json!(7)),
                "{pointer}"
            );
            assert_eq!(source.pointer(&pointer), Some(&json!(7)));
        }
    }
    let source = json!({"list":[7,8], "":{"x":true}});
    for path in [
        "/list/00", "/list/+1", "/list/-", "/list/2", "list", "/missing",
    ] {
        assert!(source.pointer(path).is_none());
    }
    assert_eq!(source.pointer("/list/1"), Some(&json!(8)));
    assert_eq!(source.pointer("//x"), Some(&json!(true)));
    assert_eq!(source.pointer(""), Some(&source));
}

#[test]
fn escaped_components_refuse_before_the_next_allocation() {
    let source = json!({"a/b":{"~key":19}});
    for at in 0..2 {
        let allocation = Refuse {
            calls: AtomicUsize::new(0),
            at,
        };
        assert_eq!(
            source.pointer_with_allocations("/a~1b/~0key", &allocation),
            Err(AllocationError::Refused)
        );
        assert_eq!(allocation.calls.load(Ordering::Relaxed), at + 1);
    }
    let allocation = Refuse {
        calls: AtomicUsize::new(0),
        at: 0,
    };
    assert!(source
        .pointer_with_allocations("/absent", &allocation)
        .unwrap()
        .is_none());
    assert_eq!(allocation.calls.load(Ordering::Relaxed), 0);
    assert_eq!(source.pointer("/a~1b/~0key"), Some(&json!(19)));
}
