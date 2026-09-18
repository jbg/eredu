//! Compare ordinary pooled validation with a separately owned funded invocation.
//! Source construction and immutable census are outside the measured interval.
use jsonschema::{OriginalValidationSource, Validator};
use std::{hint::black_box, time::Instant};
struct Invocation;
impl jsonschema::OriginalValidationFunding for Invocation {
    fn reserve(&mut self, _: usize) -> bool { true }
}
fn main() {
    let repetitions = 2_000;
    let schema = serde_json::json!({"type":"object", "patternProperties":{
        "^key_[0-9]+$":{"type":"string", "pattern":"^[a-z][a-z0-9_-]{1,32}$"}
    }, "additionalProperties":false});
    let ordinary: Validator = Validator::build_with_funding(&schema, None).unwrap();
    let graph: Validator = Validator::build_with_funding(&schema, None).unwrap();
    let prepared = OriginalValidationSource::new(graph).unwrap();
    prepared.retained_bytes().unwrap();
    for count in [1, 16, 128] {
        let mut values = serde_json::Map::new();
        for index in 0..count { values.insert(format!("key_{index}"), serde_json::json!("property_name_42")); }
        let input = serde_json::Value::Object(values);
        assert!(ordinary.is_valid(&input));
        assert!(prepared.is_valid(&input, &mut Invocation).unwrap());
        let pooled = (0..5).map(|_| {
            let start=Instant::now();
            for _ in 0..repetitions { assert!(black_box(ordinary.is_valid(black_box(&input)))); }
            start.elapsed().as_nanos()
        }).min().unwrap();
        let scoped = (0..5).map(|_| {
            let start=Instant::now();
            for _ in 0..repetitions { assert!(black_box(prepared.is_valid(black_box(&input), &mut Invocation).unwrap())); }
            start.elapsed().as_nanos()
        }).min().unwrap();
        println!("properties={count} validations={repetitions} ordinary_ns={pooled} funded_invocation_ns={scoped} ratio={:.3}",scoped as f64 / pooled as f64);
    }
}
