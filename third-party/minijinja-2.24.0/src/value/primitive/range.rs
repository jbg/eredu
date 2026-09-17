//! Exact ordinary integer progression; descriptor carries no allocation grant.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Range {
    pub(crate) start: isize,
    pub(crate) end: isize,
    pub(crate) step: Option<isize>,
    pub(crate) length: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    ZeroStep,
    TooMany,
}
impl Error {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::ZeroStep => "cannot create range with step of 0",
            Self::TooMany => "range has too many elements",
        }
    }
}
impl Range {
    pub(crate) fn prepare(
        lower: isize,
        upper: Option<isize>,
        step: Option<isize>,
    ) -> Result<Self, Error> {
        let (start, end) = upper.map_or((0, lower), |upper| (lower, upper));
        let increment = step.unwrap_or(1);
        if increment == 0 {
            return Err(Error::ZeroStep);
        }
        let distance = if (increment > 0 && start < end) || (increment < 0 && start > end) {
            start.abs_diff(end)
        } else {
            0
        };
        let width = increment.unsigned_abs();
        let length = (distance / width)
            .checked_add(usize::from(distance % width != 0))
            .ok_or(Error::TooMany)?;
        // Preserve the existing ordinary range limit, not a managed byte cap.
        if length > 100000 {
            return Err(Error::TooMany);
        }
        Ok(Self {
            start,
            end,
            step,
            length,
        })
    }
    pub(crate) fn get(self, index: usize) -> Option<isize> {
        if index >= self.length {
            return None;
        }
        let offset = i128::try_from(index)
            .ok()?
            .checked_mul(self.step.unwrap_or(1) as i128)?;
        isize::try_from((self.start as i128).checked_add(offset)?).ok()
    }
    pub(crate) fn iter(self) -> Iter {
        Iter {
            range: self,
            next: 0,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Iter {
    range: Range,
    next: usize,
}
impl Iterator for Iter {
    type Item = isize;
    fn next(&mut self) -> Option<isize> {
        let value = self.range.get(self.next)?;
        self.next += 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.range.length - self.next;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Iter {}
pub(crate) fn control_bytes() -> usize {
    use std::mem::size_of;
    size_of::<Range>()
        + size_of::<Iter>()
        + size_of::<Result<Range, Error>>()
        + size_of::<(isize, Option<isize>, Option<isize>)>()
        + size_of::<(isize, isize, isize, usize, usize, usize)>()
        + size_of::<(i128, Option<i128>, Option<isize>)>()
}
