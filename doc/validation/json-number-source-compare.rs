// Link exact local and pristine serde_json 1.0.151 builds with identical features.
extern crate serde_json as local;
extern crate serde_json_reference as reference;
use std::{hint::black_box, time::Instant};
fn main() {
    let mut inputs = vec![
        "0".to_owned(),
        "-0".into(),
        "-0.0".into(),
        "1E400".into(),
        "1e-400".into(),
        "18446744073709551616".into(),
        "01".into(),
        "1e".into(),
        "1.2e+q".into(),
        "".into(),
    ];
    let mut seed = 0x938123bcc0951a4fu64;
    for _ in 0..3000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let exponent = ((seed >> 32) % 900) as i32 - 450;
        inputs.push(format!(
            "{}{}.{}E{exponent}",
            if seed & 1 == 0 { "-" } else { "" },
            seed,
            seed.rotate_left(31)
        ));
    }
    for digits in [20, 40, 80, 160, 1000, 10000] {
        inputs.push("1".repeat(digits));
    }
    let mut comparisons = 0;
    for input in &inputs {
        let expected = reference::from_str::<reference::Value>(input)
            .map(|v| v.to_string())
            .map_err(|e| e.to_string());
        let ordinary = local::from_str::<local::Value>(input)
            .map(|v| v.to_string())
            .map_err(|e| e.to_string());
        assert_eq!(ordinary, expected, "ordinary {input}");
        comparisons += 1;
        let funded = local::bounded_events::from_slice_with_allocations(
            input.as_bytes(),
            &local::allocation::Unenforced,
        )
        .map(|v| v.to_string())
        .map_err(|e| e.to_string());
        assert_eq!(funded, expected, "events {input}");
        comparisons += 1;
        let expected = input
            .parse::<reference::Number>()
            .map(|n| {
                (
                    n.to_string(),
                    n.as_u64(),
                    n.as_i64(),
                    n.as_f64().map(f64::to_bits),
                )
            })
            .map_err(|e| e.to_string());
        let ordinary = input
            .parse::<local::Number>()
            .map(|n| {
                (
                    n.to_string(),
                    n.as_u64(),
                    n.as_i64(),
                    n.as_f64().map(f64::to_bits),
                )
            })
            .map_err(|e| e.to_string());
        assert_eq!(ordinary, expected, "number {input}");
        comparisons += 1;
        for wrapped in [
            format!("[1,{input},2]"),
            format!(
                "{{\"$serde_json::private::Number\":{}}}",
                reference::to_string(input).unwrap()
            ),
        ] {
            let expected = reference::from_str::<reference::Value>(&wrapped)
                .map(|v| v.to_string())
                .map_err(|e| e.to_string());
            let actual = local::bounded_events::from_slice_with_allocations(
                wrapped.as_bytes(),
                &local::allocation::Unenforced,
            )
            .map(|v| v.to_string())
            .map_err(|e| e.to_string());
            assert_eq!(actual, expected, "wrapped {wrapped}");
            comparisons += 1;
        }
    }
    println!("{} inputs; {comparisons} exact comparisons", inputs.len());
    for text in [
        "42",
        "1.23456789012345678901234567890",
        "[42,1.25,-3.5e12,\"text\"]",
    ] {
        let mut local_min = u128::MAX;
        let mut reference_min = u128::MAX;
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..20000 {
                black_box(local::from_str::<local::Value>(black_box(text)).unwrap());
            }
            local_min = local_min.min(start.elapsed().as_nanos());
            let start = Instant::now();
            for _ in 0..20000 {
                black_box(reference::from_str::<reference::Value>(black_box(text)).unwrap());
            }
            reference_min = reference_min.min(start.elapsed().as_nanos());
        }
        println!(
            "ordinary {text:?}: local {local_min} ns; pristine {reference_min} ns; ratio {:.3}",
            local_min as f64 / reference_min as f64
        );
    }
}
