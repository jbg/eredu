use super::*;
use eredu_nn::AttentionArithmetic;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

struct Guard {
    events: Rc<RefCell<Vec<&'static str>>>,
    live: Rc<Cell<bool>>,
    label: &'static str,
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.live.set(false);
        self.events.borrow_mut().push(self.label);
    }
}
struct Cursor {
    remaining: usize,
    _guard: Guard,
}
struct Scan {
    events: Rc<RefCell<Vec<&'static str>>>,
    block_live: Rc<Cell<bool>>,
    cursor_live: Rc<Cell<bool>>,
    fail_submit: bool,
}
impl PagedScanMechanisms for Scan {
    type Cursor = Cursor;
    type Block = Guard;
    type Output = ();
    type Error = ();
    fn begin_pass(&mut self, pass: usize) -> Result<(), ()> {
        assert!(!self.cursor_live.get());
        self.events
            .borrow_mut()
            .push(if pass == 0 { "normalize" } else { "values" });
        Ok(())
    }
    fn open_blocks(&mut self) -> Result<Cursor, ()> {
        self.cursor_live.set(true);
        Ok(Cursor {
            remaining: 2,
            _guard: Guard {
                events: self.events.clone(),
                live: self.cursor_live.clone(),
                label: "close",
            },
        })
    }
    fn next_block(&mut self, cursor: &mut Cursor) -> Result<Option<Guard>, ()> {
        assert!(!self.block_live.get());
        if cursor.remaining == 0 {
            return Ok(None);
        }
        cursor.remaining -= 1;
        self.block_live.set(true);
        self.events.borrow_mut().push("acquire");
        Ok(Some(Guard {
            events: self.events.clone(),
            live: self.block_live.clone(),
            label: "release",
        }))
    }
    fn consume_block(&mut self, _: &Guard) -> Result<(), ()> {
        assert!(self.block_live.get());
        self.events.borrow_mut().push("consume");
        Ok(())
    }
    fn submit(&mut self) -> Result<(), ()> {
        assert!(self.block_live.get());
        self.events.borrow_mut().push("submit");
        if self.fail_submit { Err(()) } else { Ok(()) }
    }
    fn consume_tail(&mut self) -> Result<(), ()> {
        assert!(!self.block_live.get());
        assert!(self.cursor_live.get());
        self.events.borrow_mut().push("tail");
        Ok(())
    }
    fn finish(&mut self) -> Result<(), ()> {
        assert!(!self.block_live.get() && !self.cursor_live.get());
        self.events.borrow_mut().push("finish");
        Ok(())
    }
}
#[test]
fn ordered_scan_holds_leases_through_submit_and_stops_before_tail_on_failure() {
    let plan = PagedScanPlan::new(
        13,
        3,
        Some(4),
        3,
        BlockwiseAttentionOptions {
            arithmetic: AttentionArithmetic::InputScores,
            softcap: Some(2.0),
        },
    )
    .unwrap();
    assert_eq!(plan.query_start(), 10);
    assert_eq!(plan.visible_start(), 7);
    assert!(plan.selects(0, 3));
    assert!(!plan.selects(3, 6));
    assert!(plan.selects(6, 9));
    assert!(!plan.selects(13, 16));
    let mut scan = Scan {
        events: Default::default(),
        block_live: Default::default(),
        cursor_live: Default::default(),
        fail_submit: true,
    };
    assert!(plan.run(&mut scan).is_err());
    assert_eq!(
        *scan.events.borrow(),
        [
            "normalize",
            "acquire",
            "consume",
            "submit",
            "release",
            "close"
        ]
    );
    scan.events.borrow_mut().clear();
    scan.fail_submit = false;
    plan.run(&mut scan).unwrap();
    assert_eq!(
        *scan.events.borrow(),
        [
            "normalize",
            "acquire",
            "consume",
            "submit",
            "release",
            "acquire",
            "consume",
            "submit",
            "release",
            "tail",
            "close",
            "values",
            "acquire",
            "consume",
            "submit",
            "release",
            "acquire",
            "consume",
            "submit",
            "release",
            "tail",
            "close",
            "finish",
        ]
    );
    assert!(PagedScanPlan::new(1, 2, None, 0, BlockwiseAttentionOptions::default()).is_err());
    assert!(PagedScanPlan::new(2, 1, Some(0), 0, BlockwiseAttentionOptions::default()).is_err());
}
