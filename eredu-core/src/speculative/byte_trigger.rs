//! Shared borrowed trigger matching. A live prefix is a length into the exact
//! immutable trigger, so token size cannot enlarge the controller's state.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Fixed prefix length into one immutable trigger. It carries no source authority.
pub struct TriggerPrefix(usize);

impl TriggerPrefix {
    /// Checks that this prefix can belong to the supplied nonempty trigger.
    pub fn is_valid_for(self, trigger: &[u8]) -> bool {
        !trigger.is_empty() && self.0 < trigger.len()
    }
    /// Borrows the prefix of the exact trigger used for previous advances.
    pub fn bytes(self, trigger: &[u8]) -> &[u8] {
        &trigger[..self.0]
    }

    /// Advances the same borrowed suffix worker without allocating.
    pub fn advance(&mut self, bytes: &[u8], trigger: &[u8]) {
        self.0 = next_prefix(self.bytes(trigger), bytes, trigger);
    }
}

/// Two borrowed pieces reproduce the former activation buffer exactly. Only
/// the grammar activation consumer materializes them; forbidden checks do not.
pub struct TriggerMatch<'a> {
    /// Trigger bytes preceding the token-local activation tail.
    pub prefix: &'a [u8],
    /// Token bytes from the activation onward.
    pub tail: &'a [u8],
    /// Whether activation starts at the beginning of this token.
    pub starts_at_token_boundary: bool,
}

/// Finds the first activation across the previous prefix and current token.
pub fn find<'a>(pending: &[u8], bytes: &'a [u8], trigger: &'a [u8]) -> Option<TriggerMatch<'a>> {
    if trigger.is_empty() {
        return None;
    }
    for start in 0..pending.len() {
        let suffix = &pending[start..];
        let Some(needed) = trigger.len().checked_sub(suffix.len()) else {
            continue;
        };
        if needed > bytes.len() || !suffix.iter().chain(bytes).take(trigger.len()).eq(trigger) {
            continue;
        }
        return Some(TriggerMatch {
            prefix: trigger,
            tail: &bytes[needed..],
            starts_at_token_boundary: false,
        });
    }
    let start = bytes
        .windows(trigger.len())
        .position(|window| window == trigger)?;
    Some(TriggerMatch {
        prefix: &[],
        tail: &bytes[start..],
        starts_at_token_boundary: start == 0,
    })
}

/// Longest suffix of the logical concatenation, without constructing it or
/// adding its lengths. This preserves overlap and arbitrary byte semantics.
pub fn next_prefix(pending: &[u8], bytes: &[u8], trigger: &[u8]) -> usize {
    (0..trigger.len())
        .rev()
        .find(|&keep| {
            if keep <= bytes.len() {
                bytes[bytes.len() - keep..] == trigger[..keep]
            } else {
                let prior = keep - bytes.len();
                prior <= pending.len()
                    && pending[pending.len() - prior..] == trigger[..prior]
                    && bytes == &trigger[prior..keep]
            }
        })
        .unwrap_or(0)
}

/// Fixed frames of the shared find/advance traversal. No allocation depends on
/// token length; the iterator and overlap state are reused in place.
pub fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<TriggerPrefix>(),
        size_of::<Option<TriggerMatch<'_>>>(),
        size_of::<(&[u8], &[u8], &[u8])>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        size_of::<std::slice::Windows<'_, u8>>(),
        size_of::<std::slice::Iter<'_, u8>>(),
        size_of::<
            std::iter::Take<std::iter::Chain<std::slice::Iter<'_, u8>, std::slice::Iter<'_, u8>>>,
        >(),
        size_of::<Option<usize>>(),
        size_of::<usize>() * 4,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
