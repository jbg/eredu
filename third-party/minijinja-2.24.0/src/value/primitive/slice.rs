//! Checked Python-style coordinates shared by ordinary and source-bound storage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    pub first: usize,
    pub stride: i128,
    pub length: usize,
}
impl Plan {
    pub fn new(length: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Option<Self> {
        if step == 0 { return None; }
        let end = length as i128;
        let reverse = step < 0;
        let lower = if reverse { -1 } else { 0 };
        let upper = if reverse { end - 1 } else { end };
        let bound = |value: Option<i64>, default| match value {
            None => default,
            Some(value) => {
                let value = i128::from(value);
                (if value < 0 { value + end } else { value }).clamp(lower, upper)
            }
        };
        let first = bound(start, if reverse { end - 1 } else { 0 });
        let stop = bound(stop, if reverse { -1 } else { end });
        let stride = i128::from(step);
        let distance = if reverse { first - stop } else { stop - first };
        let length = if distance <= 0 { 0 } else {
            usize::try_from((distance - 1) / stride.abs() + 1).ok()?
        };
        Some(Self { first: if length == 0 { 0 } else { usize::try_from(first).ok()? }, stride, length })
    }
    pub fn index(self, index: usize) -> Option<usize> {
        if index >= self.length { return None; }
        usize::try_from((self.first as i128).checked_add(self.stride.checked_mul(index as i128)?)?).ok()
    }
    pub fn compose(self, selected: Self) -> Option<Self> {
        if selected.length == 0 { return Some(Self { first:0, stride:1, length:0 }); }
        let first = self.index(selected.first)?;
        // A singleton has no second coordinate, so a large unused product is
        // neither needed nor an overflow reason.
        let stride = if selected.length == 1 { 1 } else { self.stride.checked_mul(selected.stride)? };
        let result = Self { first, stride, length:selected.length };
        self.index(selected.index(selected.length - 1)?)?;
        Some(result)
    }
    pub fn indices(self) -> impl Iterator<Item=usize> {
        (0..self.length).map(move |i| self.index(i).expect("normalized slice coordinate"))
    }
}
