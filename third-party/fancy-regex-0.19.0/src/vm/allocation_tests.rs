use super::*;
use crate::allocation::{Allocation, AllocationError};
use core::cell::Cell;
struct Funding {
    calls: Cell<usize>,
    refuse: usize,
}
impl Allocation for Funding {
    fn reserve(&self, bytes: usize) -> core::result::Result<(), AllocationError> {
        assert!(bytes > 0);
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call >= self.refuse {
            Err(AllocationError::Refused)
        } else {
            Ok(())
        }
    }
}
fn scoped_run(prog: &Prog, text: &str, allocation: Context<'_>) -> Result<Option<(usize, usize)>> {
    let mut scratch = new_scratch();
    scratch.state.reset(prog.n_saves, 0, allocation)?;
    run_inner(
        prog,
        &RegexInput::new(text),
        0,
        &HardRegexRuntimeOptions::default(),
        &mut scratch.state,
        &mut scratch.inner_slots,
        &mut [],
        &mut Caches::Fixed,
        allocation,
        |state| Ok((state.get(0), state.get(1))),
    )
}
#[test]
fn invocation_state_refuses_each_growth_without_a_pooled_allocation() {
    for (pattern, text, expected) in [
        (r"(ab)\1", "xxabab", (2, 6)),
        (r"(?:(?=a)a|(?=b)b)(a)\1", "xbaa", (1, 4)),
        (r"(a)(?>\1*)b", "xxaaaab", (2, 7)),
        (r"(a)(\1)?\1", "xaaa", (1, 4)),
    ] {
        let regex = crate::Regex::new(pattern).unwrap();
        let crate::RegexImpl::Fancy { prog, .. } = &regex.inner else {
            panic!("expected VM");
        };
        assert!(prog.body.iter().all(|insn| !matches!(
            insn,
            Insn::Delegate(_) | Insn::AbsentRepeater(_) | Insn::Seek(_)
        )));
        let funding = Funding {
            calls: Cell::new(0),
            refuse: usize::MAX,
        };
        assert_eq!(
            scoped_run(prog, text, Context::new(&funding)).unwrap(),
            Some(expected)
        );
        let requests = funding.calls.get();
        assert!(requests > 0);
        for refuse in 0..requests {
            let funding = Funding {
                calls: Cell::new(0),
                refuse,
            };
            let failure = scoped_run(prog, text, Context::new(&funding)).unwrap_err();
            assert!(
                matches!(failure, Error::Allocation(AllocationError::Refused)),
                "{} at {}: {}",
                pattern,
                refuse,
                failure
            );
            assert_eq!(funding.calls.get(), refuse + 1);
        }
    }
}
