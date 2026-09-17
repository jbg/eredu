//! Stable ordinal-decorated heap ordering, with no hidden scratch allocation.
use std::cmp::Ordering;
pub(crate) trait Driver {
    type Error;
    fn len(&self) -> usize;
    fn compare(&mut self, left: usize, right: usize) -> Result<Ordering, Self::Error>;
    fn swap(&mut self, left: usize, right: usize);
}
fn sift<D: Driver>(driver: &mut D, mut root: usize, end: usize) -> Result<(), D::Error> {
    while root < end / 2 {
        let mut child = root * 2 + 1;
        if child + 1 < end && driver.compare(child, child + 1)? == Ordering::Less {
            child += 1;
        }
        if driver.compare(root, child)? != Ordering::Less {
            break;
        }
        driver.swap(root, child);
        root = child;
    }
    Ok(())
}
pub(crate) fn run<D: Driver>(driver: &mut D) -> Result<(), D::Error> {
    let len = driver.len();
    for root in (0..len / 2).rev() {
        sift(driver, root, len)?;
    }
    for end in (1..len).rev() {
        driver.swap(0, end);
        sift(driver, 0, end)?;
    }
    Ok(())
}
pub(crate) fn ordinary(order: &mut [usize], compare: impl FnMut(usize, usize) -> Ordering) {
    struct Ordinary<'a, F> {
        order: &'a mut [usize],
        compare: F,
    }
    impl<F: FnMut(usize, usize) -> Ordering> Driver for Ordinary<'_, F> {
        type Error = std::convert::Infallible;
        fn len(&self) -> usize {
            self.order.len()
        }
        fn compare(&mut self, left: usize, right: usize) -> Result<Ordering, Self::Error> {
            let (left, right) = (self.order[left], self.order[right]);
            Ok((self.compare)(left, right).then_with(|| left.cmp(&right)))
        }
        fn swap(&mut self, left: usize, right: usize) {
            self.order.swap(left, right);
        }
    }
    match run(&mut Ordinary { order, compare }) {
        Ok(()) => (),
        Err(never) => match never {},
    }
}
pub(crate) fn control_bytes<D: Driver>() -> Option<usize> {
    use std::mem::size_of;
    size_of::<D>()
        .checked_add(size_of::<[&mut D; 2]>())?
        .checked_add(size_of::<[usize; 7]>())?
        .checked_add(size_of::<[std::iter::Rev<std::ops::Range<usize>>; 2]>())?
        .checked_add(size_of::<Result<Ordering, D::Error>>())?
        .checked_add(size_of::<Result<(), D::Error>>())
}
