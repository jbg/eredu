//! Borrowed positive-stride destination algebra shared by spatial and temporal prefixes.
use super::*;

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub(crate) enum PrefixError {
    #[error("prefix destination ranks differ")]
    Rank,
    #[error("prefix destination exceeds selected shape")]
    Extent,
    #[error("prefix destination arithmetic overflow")]
    Overflow,
}
impl From<PrefixError> for CaptureError {
    fn from(value: PrefixError) -> Self {
        match value {
            PrefixError::Overflow => Self::Overflow,
            value => Self::Invalid(value.to_string()),
        }
    }
}
pub(crate) struct PrefixDestination<'a> {
    global: &'a [u64],
    local: &'a [u64],
    starts: &'a [u64],
    strides: &'a [u64],
    count: u64,
}
impl<'a> PrefixDestination<'a> {
    pub(super) fn new(
        global: &'a [u64],
        local: &'a [u64],
        starts: &'a [u64],
        strides: &'a [u64],
    ) -> Result<Self, CaptureError> {
        Self::new_fixed(global, local, starts, strides).map_err(Into::into)
    }
    pub(crate) fn new_fixed(
        global: &'a [u64],
        local: &'a [u64],
        starts: &'a [u64],
        strides: &'a [u64],
    ) -> Result<Self, PrefixError> {
        if [local.len(), starts.len(), strides.len()]
            .iter()
            .any(|n| *n != global.len())
        {
            return Err(PrefixError::Rank);
        }
        elements(global).map_err(|_| PrefixError::Overflow)?;
        let count = elements(local).map_err(|_| PrefixError::Overflow)?;
        for i in 0..global.len() {
            if strides[i] == 0
                || starts[i] > global[i]
                || (local[i] != 0
                    && add(
                        starts[i],
                        mul(local[i] - 1, strides[i]).map_err(|_| PrefixError::Overflow)?,
                    )
                    .map_err(|_| PrefixError::Overflow)?
                        >= global[i])
            {
                return Err(PrefixError::Extent);
            }
        }
        Ok(Self {
            global,
            local,
            starts,
            strides,
            count,
        })
    }
    /// The checked positive-stride mapping is an increasing injection. Thus its
    /// local prefix N contains every mapped global selected ordinal below N.
    pub(crate) fn ordinal(&self, local: u64) -> Option<u64> {
        if local >= self.count {
            return None;
        }
        let mut remainder = local;
        let mut output = 0;
        let mut stride = 1;
        for axis in (0..self.global.len()).rev() {
            let coordinate = remainder % self.local[axis];
            remainder /= self.local[axis];
            output += (self.starts[axis] + coordinate * self.strides[axis]) * stride;
            stride *= self.global[axis];
        }
        Some(output)
    }
}
