//! Checked source-global coordinate equations shared by legacy and fixed storage.
use crate::media_plan::MediaSemanticError;

#[derive(Clone, Copy)]
pub(crate) enum GridRows<'a> {
    Tuples(&'a [(i32, i32, i32)]),
    Arrays(&'a [[i32; 3]]),
}
impl<'a> GridRows<'a> {
    pub(crate) fn iter(self) -> impl ExactSizeIterator<Item = (i32, i32, i32)> + Clone + 'a {
        (0..self.len()).map(move |index| {
            let [t, h, w] = self.row(index);
            (t, h, w)
        })
    }
    fn len(self) -> usize {
        match self {
            Self::Tuples(v) => v.len(),
            Self::Arrays(v) => v.len(),
        }
    }
    fn row(self, i: usize) -> [i32; 3] {
        match self {
            Self::Tuples(v) => {
                let (a, b, c) = v[i];
                [a, b, c]
            }
            Self::Arrays(v) => v[i],
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) enum PositionComponent<'a> {
    Text(i64),
    Media(GridRows<'a>),
}
pub(crate) struct PositionDestination<'a> {
    pub axes: [&'a mut [i32]; 3],
    pub prefix: Option<&'a mut [i32]>,
}
struct Cursor<'a> {
    current: i64,
    count: usize,
    maximum: i64,
    destination: Option<PositionDestination<'a>>,
}
fn overflow() -> MediaSemanticError {
    MediaSemanticError::overflow("Qwen multimodal position overflow")
}
impl Cursor<'_> {
    fn validate_add(
        &self,
        count: u64,
        maximum: i64,
        next: i64,
    ) -> Result<usize, MediaSemanticError> {
        if maximum > i64::from(i32::MAX) || next > i64::from(i32::MAX) {
            return Err(overflow());
        }
        let count = usize::try_from(count).map_err(|_| overflow())?;
        let total = self.count.checked_add(count).ok_or_else(overflow)?;
        if total > i32::MAX as usize {
            return Err(overflow());
        }
        Ok(total)
    }
    fn write(&mut self, values: [i64; 3]) {
        for v in values {
            self.maximum = self.maximum.max(v);
        }
        if let Some(destination) = &mut self.destination {
            for (axis, value) in destination.axes.iter_mut().zip(values) {
                axis[self.count] = value as i32;
            }
            if let Some(prefix) = &mut destination.prefix {
                prefix[self.count] = (self.maximum - self.count as i64) as i32;
            }
        }
        self.count += 1;
    }
    fn text(&mut self, length: i64) -> Result<(), MediaSemanticError> {
        if length <= 0 {
            return Err(MediaSemanticError::input("empty or invalid position part"));
        }
        let next = self.current.checked_add(length).ok_or_else(overflow)?;
        let total = self.validate_add(length as u64, next - 1, next)?;
        if self.destination.is_some() {
            for value in self.current..next {
                self.write([value; 3]);
            }
        } else {
            self.count = total;
            self.maximum = self.maximum.max(next - 1);
        }
        self.current = next;
        Ok(())
    }
    fn media(&mut self, rows: GridRows<'_>, merge: i32) -> Result<(), MediaSemanticError> {
        if rows.len() == 0 {
            return Err(MediaSemanticError::input("empty or invalid position part"));
        }
        for i in 0..rows.len() {
            let [t, h, w] = rows.row(i);
            if t <= 0 || h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0 {
                return Err(MediaSemanticError::input("invalid merged media grid"));
            }
            let (t, h, w) = (i64::from(t), i64::from(h / merge), i64::from(w / merge));
            let count = (t as u64)
                .checked_mul(h as u64)
                .and_then(|n| n.checked_mul(w as u64))
                .ok_or_else(overflow)?;
            let maximum = self
                .current
                .checked_add(t.max(h).max(w) - 1)
                .ok_or_else(overflow)?;
            // Preserve the released equation: time is not part of this advance.
            let next = self.current.checked_add(h.max(w)).ok_or_else(overflow)?;
            let total = self.validate_add(count, maximum, next)?;
            if self.destination.is_some() {
                for temporal in 0..t {
                    for y in 0..h {
                        for x in 0..w {
                            self.write([
                                self.current + temporal,
                                self.current + y,
                                self.current + x,
                            ]);
                        }
                    }
                }
            } else {
                self.count = total;
                self.maximum = self.maximum.max(maximum);
            }
            self.current = next;
        }
        Ok(())
    }
}
fn walk<'a>(
    parts: impl Iterator<Item = Result<PositionComponent<'a>, MediaSemanticError>>,
    merge: i32,
    expected: usize,
    destination: Option<PositionDestination<'_>>,
) -> Result<i32, MediaSemanticError> {
    if merge <= 0 || expected == 0 || expected > i32::MAX as usize {
        return Err(MediaSemanticError::input(
            "multimodal positions require positive geometry",
        ));
    }
    let mut cursor = Cursor {
        current: 0,
        count: 0,
        maximum: -1,
        destination,
    };
    let mut seen = false;
    for part in parts {
        seen = true;
        match part? {
            PositionComponent::Text(n) => cursor.text(n)?,
            PositionComponent::Media(rows) => cursor.media(rows, merge)?,
        };
    }
    if !seen || cursor.count != expected {
        return Err(MediaSemanticError::input(
            "position metadata differs from expected token count",
        )
        .scalars(expected as u64, cursor.count as i128));
    }
    i32::try_from(cursor.maximum + 1 - cursor.count as i64).map_err(|_| overflow())
}
/// Private equation adapter: both callers use immutable source-backed iterators.
/// It performs scalar/grid work only, even for an enormous text length.
pub(crate) fn validate_positions<'a>(
    parts: impl Iterator<Item = Result<PositionComponent<'a>, MediaSemanticError>>,
    merge: i32,
    expected: usize,
) -> Result<i32, MediaSemanticError> {
    walk(parts, merge, expected, None)
}
pub(crate) fn emit_positions<'a>(
    parts: impl Iterator<Item = Result<PositionComponent<'a>, MediaSemanticError>> + Clone,
    merge: i32,
    expected: usize,
    destination: PositionDestination<'_>,
) -> Result<i32, MediaSemanticError> {
    // No write precedes the complete overflow/count validation, including a bad
    // later part. No user callback can obtain this crate-private fixed emitter.
    validate_positions(parts.clone(), merge, expected)?;
    if destination.axes.iter().any(|axis| axis.len() != expected)
        || destination
            .prefix
            .as_ref()
            .is_some_and(|p| p.len() != expected)
    {
        return Err(MediaSemanticError::input(
            "position destination extent mismatch",
        ));
    }
    walk(parts, merge, expected, Some(destination))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn late_position_overflow_rejects_without_touching_any_destination() {
        let parts = [
            PositionComponent::Text(i64::from(i32::MAX)),
            PositionComponent::Text(1),
        ];
        let mut a = [71; 2];
        let mut b = [72; 2];
        let mut c = [73; 2];
        let mut prefix = [74; 2];
        let error = emit_positions(
            parts.into_iter().map(Ok),
            1,
            2,
            PositionDestination {
                axes: [&mut a, &mut b, &mut c],
                prefix: Some(&mut prefix),
            },
        )
        .unwrap_err();
        assert_eq!(error, overflow());
        assert_eq!(a, [71; 2]);
        assert_eq!(b, [72; 2]);
        assert_eq!(c, [73; 2]);
        assert_eq!(prefix, [74; 2]);
    }
    #[test]
    fn temporal_extent_does_not_change_spatial_advance_and_every_prefix_is_exact() {
        let grid = [[5, 2, 2]];
        let parts = [
            PositionComponent::Text(2),
            PositionComponent::Media(GridRows::Arrays(&grid)),
            PositionComponent::Text(2),
        ];
        let mut a = [0; 9];
        let mut b = [0; 9];
        let mut c = [0; 9];
        let mut p = [0; 9];
        let delta = emit_positions(
            parts.into_iter().map(Ok),
            2,
            9,
            PositionDestination {
                axes: [&mut a, &mut b, &mut c],
                prefix: Some(&mut p),
            },
        )
        .unwrap();
        assert_eq!(a, [0, 1, 2, 3, 4, 5, 6, 3, 4]);
        assert_eq!(b, [0, 1, 2, 2, 2, 2, 2, 3, 4]);
        assert_eq!(c, b);
        assert_eq!(p, [0, 0, 0, 0, 0, 0, 0, -1, -2]);
        assert_eq!(delta, -2);
    }
}
