extern crate fraction;
use fraction::{
    operations::{Allocation, AllocationError, BigUintOperations},
    BigFraction,
};
use std::cell::Cell;
struct Account {
    calls: Cell<usize>,
    fail: Option<usize>,
    refused: Cell<bool>,
}
impl Account {
    fn new(fail: Option<usize>) -> Self {
        Self {
            calls: Cell::new(0),
            fail,
            refused: Cell::new(false),
        }
    }
}
impl Allocation for Account {
    fn reserve(&self, _bytes: usize) -> Result<(), AllocationError> {
        assert!(!self.refused.get(), "producer called after first refusal");
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.fail == Some(call) {
            self.refused.set(true);
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}
fn quotient(left: f64, right: f64, account: &Account) -> Result<BigFraction, AllocationError> {
    let operations = BigUintOperations(account);
    let left = BigFraction::from_f64_with_operations(left, &operations)?;
    let right = BigFraction::from_f64_with_operations(right, &operations)?;
    left.div_with_operations(right, &operations)
}
#[test]
fn original_float_and_ratio_producers_stop_at_each_reached_refusal() {
    for (left, right) in [
        (0.0, 0.1),
        (1.2, 0.1),
        (1070468.14, 0.01),
        (1e300, 0.1),
        (f64::MAX, 0.7),
        (1e-15, 1e-17),
        (f64::MIN_POSITIVE, 0.1),
        (-1e300, 0.123456789),
        (f64::from_bits(1), 1e-300),
        (2.2250738585072014e-308, 1.7976931348623157e308),
    ] {
        let expected = BigFraction::from(left) / BigFraction::from(right);
        let account = Account::new(None);
        assert_eq!(
            quotient(left, right, &account).unwrap(),
            expected,
            "{left}/{right}"
        );
        for call in 0..account.calls.get() {
            let account = Account::new(Some(call));
            assert_eq!(
                quotient(left, right, &account),
                Err(AllocationError::Refused),
                "{left}/{right} at{call}"
            );
            assert_eq!(account.calls.get(), call + 1);
        }
    }
}
