//! In-place ordering with constant control storage and no recursive library sort.
use crate::util::allocation::{AllocationError, Allocator};
use core::{cmp::Ordering, mem::size_of};

struct Control {
    root: usize,
    end: usize,
    child: usize,
    selected: usize,
}

pub(in crate::dfa) fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(
    values: &mut [T],
    mut compare: F,
    allocation: Allocator<'_>,
) -> Result<(), AllocationError> {
    allocation.reserve(size_of::<Control>() + size_of::<F>())?;
    let mut control = Control {
        root: 0,
        end: values.len(),
        child: 0,
        selected: 0,
    };
    // Build a maximum heap bottom-up, then extract its largest element.
    for root in (0..values.len() / 2).rev() {
        control.root = root;
        sift(values, &mut compare, &mut control);
    }
    while control.end > 1 {
        control.end -= 1;
        values.swap(0, control.end);
        control.root = 0;
        sift(values, &mut compare, &mut control);
    }
    Ok(())
}
fn sift<T, F: FnMut(&T, &T) -> Ordering>(values: &mut [T], compare: &mut F, control: &mut Control) {
    while control.root < control.end / 2 {
        control.child = control.root * 2 + 1;
        control.selected = control.child;
        if control.child + 1 < control.end
            && compare(&values[control.child], &values[control.child + 1]) == Ordering::Less
        {
            control.selected += 1;
        }
        if compare(&values[control.root], &values[control.selected]) != Ordering::Less {
            break;
        }
        values.swap(control.root, control.selected);
        control.root = control.selected;
    }
}
