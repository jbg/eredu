//! Standalone oracle linking the workspace fraction worker and pristine pinned
//! fraction/num-rational/num-bigint archives as `fraction_reference`.
use std::{cell::Cell, hint::black_box, time::Instant};
struct Account(Cell<usize>);
impl fraction::operations::Allocation for Account {
    fn reserve(&self, _: usize) -> Result<(), fraction::operations::AllocationError> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}
fn main() {
    let mut inputs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        0.7,
        1070468.14,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::EPSILON,
    ];
    for exponent in -324..=308 {
        inputs.push(10.0_f64.powi(exponent));
    }
    let mut seed = 0x749a_2435_6141_aedd_u64;
    for _ in 0..8192 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let value = f64::from_bits(seed);
        if value.is_finite() {
            inputs.push(value);
        }
    }
    let account = Account(Cell::new(0));
    let operations = fraction::operations::BigUintOperations(&account);
    let mut comparisons = 0;
    for (index, &value) in inputs.iter().enumerate() {
        let local = fraction::BigFraction::from(value);
        let reference = fraction_reference::BigFraction::from(value);
        assert_eq!(
            format!("{local:?}"),
            format!("{reference:?}"),
            "source {value:?}"
        );
        comparisons += 1;
        let paid = fraction::BigFraction::from_f64_with_operations(value, &operations).unwrap();
        assert_eq!(
            format!("{paid:?}"),
            format!("{reference:?}"),
            "paid source {value:?}"
        );
        comparisons += 1;
        let other = inputs[(index * 17 + 3) % inputs.len()];
        let local = local / fraction::BigFraction::from(other);
        let reference = reference / fraction_reference::BigFraction::from(other);
        assert_eq!(
            format!("{local:?}"),
            format!("{reference:?}"),
            "division {value:?}/{other:?}"
        );
        comparisons += 1;
        let paid = paid
            .div_with_operations(
                fraction::BigFraction::from_f64_with_operations(other, &operations).unwrap(),
                &operations,
            )
            .unwrap();
        assert_eq!(
            format!("{paid:?}"),
            format!("{reference:?}"),
            "paid division {value:?}/{other:?}"
        );
        comparisons += 1;
    }
    println!("{comparisons} exact ordinary/paid fraction and quotient comparisons across{} sources passed;{} reached reservations",inputs.len(),account.0.get());
    for (name, left, right) in [
        ("decimal", 1070468.14, 0.01),
        ("large", 1e300, 0.7),
        ("tiny", 1e-200, 0.123456789),
    ] {
        let count = 2000;
        let start = Instant::now();
        for _ in 0..count {
            black_box(
                fraction_reference::BigFraction::from(black_box(left))
                    / fraction_reference::BigFraction::from(black_box(right)),
            );
        }
        let prior = start.elapsed();
        let start = Instant::now();
        for _ in 0..count {
            black_box(
                fraction::BigFraction::from(black_box(left))
                    / fraction::BigFraction::from(black_box(right)),
            );
        }
        let current = start.elapsed();
        println!(
            "{name} {count} ordinary calls: local={current:?} pristine={prior:?} ratio={:.3}",
            current.as_secs_f64() / prior.as_secs_f64()
        );
    }
}
