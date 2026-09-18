use super::*;
use crate::{Draft, Error, Registry, ResourceRef};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Funding {
    calls: AtomicUsize,
    bytes: AtomicUsize,
    stop: usize,
}
impl Funding {
    fn new(stop: usize) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            stop,
        }
    }
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if call == self.stop {
            return Err(AllocationError::Refused);
        }
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
}

fn schema() -> Value {
    json!({
        "$id":"https://example.com/root",
        "$defs": {
            "a/b": {"$anchor":"first", "type":"string"},
            "c~d": {"$anchor":"second", "type":"number"},
            "third": {"$anchor":"third", "type":"boolean"},
            "nested": {"$id":"child", "$dynamicAnchor":"node", "type":"integer"}
        },
        "allOf":[{"$ref":"#/$defs/a~1b"},{"$ref":"#/$defs/c~0d"}]
    })
}
fn exercise(funding: &dyn Allocation, value: &Value, owned: Option<Value>) -> Result<(), Error> {
    let builder = Registry::new_with_allocations(funding)?.draft(Draft::Draft202012);
    let registry = match owned {
        Some(value) => builder.add("https://example.com/root", value)?.prepare()?,
        None => builder
            .add(
                "https://example.com/root",
                ResourceRef::new(value, Draft::Draft202012),
            )?
            .prepare()?,
    };
    let resolver = registry.try_resolver(crate::uri::from_str_with_allocations(
        "https://example.com/root",
        funding,
    )?)?;
    assert_eq!(resolver.lookup("#first")?.contents()["type"], "string");
    assert_eq!(
        resolver.lookup("#/$defs/c~0d")?.contents()["type"],
        "number"
    );
    assert_eq!(resolver.lookup("child#node")?.contents()["type"], "integer");
    Ok(())
}
#[test]
fn refuses_every_reached_borrowed_and_owned_registry_allocation() {
    let value = schema();
    for owned in [false, true] {
        let success = Funding::new(usize::MAX);
        exercise(&success, &value, owned.then(|| value.clone())).unwrap();
        let count = success.calls.load(Ordering::Relaxed);
        assert!(count > 20);
        for stop in 0..count {
            let funding = Funding::new(stop);
            // This source copy belongs to the caller and is complete before preparation starts.
            let owned_value = owned.then(|| value.clone());
            let result = exercise(&funding, &value, owned_value);
            assert!(
                matches!(result, Err(Error::Allocation(AllocationError::Refused))),
                "stop {stop}/{count}, owned={owned}: {result:?}"
            );
            assert_eq!(
                funding.calls.load(Ordering::Relaxed),
                stop + 1,
                "continued after refusal {stop}"
            );
        }
    }
}
#[test]
fn escaped_pointer_and_error_diagnostics_preserve_refusal() {
    let value = schema();
    for reference in ["#missing", "#/%FF", "#/$defs/missing", "#/$defs/a%2Fb"] {
        let success = Funding::new(usize::MAX);
        let registry = Registry::new_with_allocations(&success)
            .unwrap()
            .add("https://example.com/root", &value)
            .unwrap()
            .prepare()
            .unwrap();
        let resolver = registry
            .try_resolver(crate::uri::from_str("https://example.com/root").unwrap())
            .unwrap();
        let before = success.calls.load(Ordering::Relaxed);
        let expected = resolver.lookup(reference).unwrap_err().to_string();
        let total = success.calls.load(Ordering::Relaxed);
        assert!(total > before);
        for stop in before..total {
            let funding = Funding::new(stop);
            let registry = Registry::new_with_allocations(&funding)
                .unwrap()
                .add("https://example.com/root", &value)
                .unwrap()
                .prepare()
                .unwrap();
            let resolver = registry
                .try_resolver(crate::uri::from_str("https://example.com/root").unwrap())
                .unwrap();
            let error = resolver.lookup(reference).unwrap_err();
            assert!(
                matches!(error, Error::Allocation(AllocationError::Refused)),
                "{reference}: {expected}: {error}"
            );
            assert_eq!(funding.calls.load(Ordering::Relaxed), stop + 1);
        }
    }
}
#[test]
fn small_map_promotion_refusal_preserves_existing_entries() {
    let mut map: crate::small_map::SmallMap<u32, u32> = crate::small_map::SmallMap::new();
    map.insert(1, 10);
    map.insert(2, 20);
    assert_eq!(
        map.try_insert(3, 30, Allocator(&Funding::new(0))),
        Err(AllocationError::Refused)
    );
    assert_eq!(map.get(&1), Some(&10));
    assert_eq!(map.get(&2), Some(&20));
    assert_eq!(map.get(&3), None);
}

#[test]
fn opaque_retrieval_is_refused_before_callback() {
    struct External(AtomicUsize);
    impl crate::Retrieve for External {
        fn retrieve(
            &self,
            _: &crate::Uri<String>,
        ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Value::Bool(true))
        }
    }
    let source = json!({"$ref":"https://remote.example/schema"});
    let external = Arc::new(External(AtomicUsize::new(0)));
    let retriever: Arc<dyn crate::Retrieve> = external.clone();
    let funding = Funding::new(usize::MAX);
    let result = Registry::new_with_allocations(&funding)
        .unwrap()
        .try_retriever(retriever.clone())
        .unwrap()
        .add("https://example.com/root", &source)
        .unwrap()
        .prepare();
    assert!(matches!(
        result,
        Err(Error::Unqualified("external retriever"))
    ));
    assert_eq!(external.0.load(Ordering::Relaxed), 0);
    Registry::new()
        .retriever(retriever)
        .add("https://example.com/root", &source)
        .unwrap()
        .prepare()
        .unwrap();
    assert_eq!(external.0.load(Ordering::Relaxed), 1);
}

#[test]
fn deep_deferred_crawl_refuses_without_retry() {
    let mut source = json!({"$anchor":"leaf","type":"string"});
    for _ in 0..140 {
        let mut object = serde_json::Map::new();
        object.insert("allOf".to_owned(), Value::Array(vec![source]));
        source = Value::Object(object);
    }
    let success = Funding::new(usize::MAX);
    Registry::new_with_allocations(&success)
        .unwrap()
        .add("https://example.com/deep", &source)
        .unwrap()
        .prepare()
        .unwrap();
    let total = success.calls.load(Ordering::Relaxed);
    for stop in 0..total {
        let funding = Funding::new(stop);
        let result = (|| {
            Registry::new_with_allocations(&funding)?
                .add("https://example.com/deep", &source)?
                .prepare()
        })();
        assert!(
            matches!(result, Err(Error::Allocation(AllocationError::Refused))),
            "deep refusal {stop}/{total}"
        );
        assert_eq!(funding.calls.load(Ordering::Relaxed), stop + 1);
    }
}

#[test]
fn bundled_meta_sources_are_owned_and_refuse_each_allocation() {
    let source = json!({"$ref":"https://json-schema.org/draft/2020-12/schema"});
    let run = |funding: &dyn Allocation| -> Result<(), Error> {
        let registry = Registry::new_with_allocations(funding)?
            .add("https://example.com/root", &source)?
            .prepare()?;
        let resolver = registry.try_resolver(crate::uri::from_str_with_allocations(
            "https://example.com/root",
            funding,
        )?)?;
        assert_eq!(
            resolver
                .lookup("https://json-schema.org/draft/2020-12/schema")?
                .contents()["$id"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        Ok(())
    };
    let success = Funding::new(usize::MAX);
    run(&success).unwrap();
    let count = success.calls.load(Ordering::Relaxed);
    assert!(count > 100);
    for stop in 0..count {
        let funding = Funding::new(stop);
        let result = run(&funding);
        assert!(
            matches!(result, Err(Error::Allocation(AllocationError::Refused))),
            "bundled producer {stop}/{count}: {result:?}"
        );
        assert_eq!(funding.calls.load(Ordering::Relaxed), stop + 1);
    }
    eprintln!(
        "bundled registry: {count} reached requests, {} cumulative requested bytes",
        success.bytes.load(Ordering::Relaxed)
    );
}

#[test]
fn deep_reference_target_outside_schema_keywords_uses_paid_work_list() {
    let mut target = Value::Bool(true);
    for _ in 0..5000 {
        let mut object = serde_json::Map::new();
        object.insert("allOf".to_owned(), Value::Array(vec![target]));
        target = Value::Object(object);
    }
    let mut root = serde_json::Map::new();
    root.insert("$ref".to_owned(), Value::String("#/component".to_owned()));
    root.insert("component".to_owned(), target);
    let source = Value::Object(root);
    let funding = Funding::new(usize::MAX);
    let registry = Registry::new_with_allocations(&funding)
        .unwrap()
        .add("https://example.com/root", &source)
        .unwrap()
        .prepare()
        .unwrap();
    assert!(registry
        .try_contains_resource("https://example.com/root")
        .unwrap());
    drop(registry);
    // Value drop is recursive; this fixture isolates the registry's traversal contract.
    std::mem::forget(source);
}
