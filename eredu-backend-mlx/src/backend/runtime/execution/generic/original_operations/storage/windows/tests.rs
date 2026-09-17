use super::*;
use std::{cell::RefCell, rc::Rc};

#[test]
fn ordinal_banks_keep_shapes_and_finite_cursors_through_warm_miss_omissions() {
    let mut windows = [2usize, 7, 3]
        .into_iter()
        .enumerate()
        .map(|(index, rows)| {
            PreparedOperationBank::try_new(2, move |attempt| {
                Ok::<_, Infallible>((index, attempt, rows))
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    // Independent expected schedule: window 2 is visited twice while 0 and 1
    // remain at their own cursor. Warm/miss choice consumes local slots only.
    assert_eq!(checkout(&mut windows, 2).unwrap(), (2, 0, 3));
    assert_eq!(checkout(&mut windows, 0).unwrap(), (0, 0, 2));
    assert_eq!(checkout(&mut windows, 2).unwrap(), (2, 1, 3));
    assert_eq!(
        checkout(&mut windows, 2),
        Err(WindowCheckoutFailure::Exhausted)
    );
    assert_eq!(
        checkout(&mut windows, 99),
        Err(WindowCheckoutFailure::UnknownWindow)
    );
    assert_eq!(checkout(&mut windows, 1).unwrap(), (1, 0, 7));
    assert_eq!(checkout(&mut windows, 0).unwrap(), (0, 1, 2));
    assert_eq!(checkout(&mut windows, 1).unwrap(), (1, 1, 7));
    for index in 0..3 {
        assert_eq!(
            checkout(&mut windows, index),
            Err(WindowCheckoutFailure::Exhausted)
        );
    }
}

#[derive(Debug)]
struct Slot {
    label: (usize, usize, usize),
    drops: Rc<RefCell<Vec<(usize, usize, usize)>>>,
}
impl Drop for Slot {
    fn drop(&mut self) {
        self.drops.borrow_mut().push(self.label);
    }
}
#[derive(Debug)]
struct Attempt {
    warm: Option<Slot>,
    miss: Option<Slot>,
}
#[test]
fn dropped_attempt_retires_unused_slots_while_issued_payload_survives_other_windows() {
    let drops = Rc::new(RefCell::new(Vec::new()));
    let mut windows = (0..2)
        .map(|index| {
            let drops = drops.clone();
            PreparedOperationBank::try_new(2, move |attempt| {
                Ok::<_, Infallible>(Attempt {
                    warm: Some(Slot {
                        label: (index, attempt, 0),
                        drops: drops.clone(),
                    }),
                    miss: Some(Slot {
                        label: (index, attempt, 1),
                        drops: drops.clone(),
                    }),
                })
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut first = checkout(&mut windows, 1).unwrap();
    let issued = first.warm.take().unwrap();
    drop(first); // Warm hit leaves only its unused miss slot to retire.
    assert_eq!(&*drops.borrow(), &[(1, 0, 1)]);
    let cancelled = checkout(&mut windows, 0).unwrap();
    drop(cancelled); // Cancellation before issue retires both local slots.
    assert_eq!(&*drops.borrow(), &[(1, 0, 1), (0, 0, 0), (0, 0, 1)]);
    let next = checkout(&mut windows, 1).unwrap();
    assert_eq!(next.warm.as_ref().unwrap().label, (1, 1, 0));
    assert_eq!(next.miss.as_ref().unwrap().label, (1, 1, 1));
    drop((next, windows));
    assert!(!drops.borrow().contains(&(1, 0, 0)));
    assert_eq!(drops.borrow().len(), 7);
    drop(issued);
    assert_eq!(drops.borrow().last(), Some(&(1, 0, 0)));
    assert_eq!(drops.borrow().len(), 8);
}
