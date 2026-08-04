//! Architecture-neutral decoder layer schedules and attention geometry.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, num::NonZeroU32};

/// Attention behavior for one decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionPolicy {
    /// Attend to the complete causal prefix.
    Full,
    /// Attend to at most `window` positions, including the current token.
    Sliding {
        /// Exact positive number of visible positions.
        window: NonZeroU32,
    },
}

impl AttentionPolicy {
    /// Creates a sliding policy from a positive window.
    pub fn sliding(window: u32) -> Result<Self, LayerScheduleError> {
        let window = NonZeroU32::new(window).ok_or(LayerScheduleError::ZeroWindow)?;
        Ok(Self::Sliding { window })
    }

    /// Returns the sliding window, or `None` for full attention.
    pub const fn window(self) -> Option<NonZeroU32> {
        match self {
            Self::Full => None,
            Self::Sliding { window } => Some(window),
        }
    }
}

/// Validated, ordered policy for every decoder layer.
///
/// The policy type is architecture-defined. Pure-attention decoders use
/// `LayerSchedule<AttentionPolicy>`; hybrid decoders can use their own policy
/// enum without weakening layer-count validation or indexed access.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerSchedule<P> {
    layers: Box<[P]>,
}

impl<P> LayerSchedule<P> {
    /// Validates an exact ordered policy list against the decoder layer count.
    pub fn new(layer_count: usize, layers: Vec<P>) -> Result<Self, LayerScheduleError> {
        if layer_count == 0 {
            return Err(LayerScheduleError::Empty);
        }
        if layers.len() != layer_count {
            return Err(LayerScheduleError::LayerCount {
                expected: layer_count,
                actual: layers.len(),
            });
        }
        Ok(Self {
            layers: layers.into_boxed_slice(),
        })
    }

    /// Returns the number of decoder layers represented by the schedule.
    pub const fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns whether the schedule contains no layers.
    pub const fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns one layer policy, with out-of-range indices reported as `None`.
    pub fn get(&self, layer: usize) -> Option<&P> {
        self.layers.get(layer)
    }

    /// Iterates over policies in architecture layer order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &P> + '_ {
        self.layers.iter()
    }
}

impl LayerSchedule<AttentionPolicy> {
    /// Creates an all-full attention schedule.
    pub fn all_full(layer_count: usize) -> Result<Self, LayerScheduleError> {
        Self::new(layer_count, vec![AttentionPolicy::Full; layer_count])
    }

    /// Creates an all-sliding attention schedule with one exact positive window.
    pub fn all_sliding(layer_count: usize, window: u32) -> Result<Self, LayerScheduleError> {
        let policy = AttentionPolicy::sliding(window)?;
        Self::new(layer_count, vec![policy; layer_count])
    }

    /// Creates an attention schedule from a Boolean pattern where `true` means sliding.
    pub fn from_sliding_pattern(
        layer_count: usize,
        pattern: &[bool],
        window: Option<u32>,
    ) -> Result<Self, LayerScheduleError> {
        if pattern.len() != layer_count {
            return Err(LayerScheduleError::LayerCount {
                expected: layer_count,
                actual: pattern.len(),
            });
        }
        let sliding = pattern.iter().any(|enabled| *enabled);
        let policy = match (sliding, window) {
            (true, Some(window)) => Some(AttentionPolicy::sliding(window)?),
            (true, None) => return Err(LayerScheduleError::MissingWindow),
            (false, _) => None,
        };
        Self::new(
            layer_count,
            pattern
                .iter()
                .map(|enabled| {
                    if *enabled {
                        policy.expect("validated enabled pattern")
                    } else {
                        AttentionPolicy::Full
                    }
                })
                .collect(),
        )
    }

    /// Returns the number of full-attention layers.
    pub fn full_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|policy| matches!(policy, AttentionPolicy::Full))
            .count()
    }

    /// Returns the number of sliding-attention layers.
    pub fn sliding_layer_count(&self) -> usize {
        self.len() - self.full_layer_count()
    }

    /// Counts sliding layers by exact window in deterministic order.
    pub fn sliding_windows(&self) -> BTreeMap<NonZeroU32, usize> {
        let mut windows = BTreeMap::new();
        for window in self.iter().copied().filter_map(AttentionPolicy::window) {
            *windows.entry(window).or_default() += 1;
        }
        windows
    }

    /// Returns a stable ordered representation suitable for fingerprints.
    pub fn fingerprint_component(&self) -> String {
        self.iter()
            .copied()
            .map(|policy| match policy {
                AttentionPolicy::Full => "f".to_string(),
                AttentionPolicy::Sliding { window } => format!("s{}", window.get()),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Validation error for an ordered per-layer policy schedule.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LayerScheduleError {
    /// A decoder-layer schedule cannot be empty.
    Empty,
    /// The supplied ordered policy count differs from the decoder depth.
    LayerCount {
        /// Decoder layer count.
        expected: usize,
        /// Supplied policy count.
        actual: usize,
    },
    /// A sliding window was zero.
    ZeroWindow,
    /// At least one layer enables sliding attention but no window was supplied.
    MissingWindow,
}

impl fmt::Display for LayerScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("layer schedule must contain at least one layer"),
            Self::LayerCount { expected, actual } => write!(
                formatter,
                "layer schedule has {actual} entries for {expected} decoder layers"
            ),
            Self::ZeroWindow => formatter.write_str("sliding attention window must be positive"),
            Self::MissingWindow => formatter
                .write_str("sliding attention is enabled for at least one layer without a window"),
        }
    }
}

impl std::error::Error for LayerScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_preserve_arbitrary_order_and_distinct_windows() {
        let schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(4).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(8).unwrap(),
                AttentionPolicy::sliding(4).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(schedule.full_layer_count(), 1);
        assert_eq!(schedule.sliding_layer_count(), 3);
        assert_eq!(
            schedule.sliding_windows().into_iter().collect::<Vec<_>>(),
            vec![
                (NonZeroU32::new(4).unwrap(), 2),
                (NonZeroU32::new(8).unwrap(), 1)
            ]
        );
        assert_eq!(schedule.fingerprint_component(), "s4,f,s8,s4");
    }

    #[test]
    fn pattern_validation_is_exact() {
        assert!(matches!(
            LayerSchedule::from_sliding_pattern(2, &[true], Some(4)),
            Err(LayerScheduleError::LayerCount { .. })
        ));
        assert_eq!(
            LayerSchedule::from_sliding_pattern(2, &[true, false], None),
            Err(LayerScheduleError::MissingWindow)
        );
        assert!(AttentionPolicy::sliding(0).is_err());
    }

    #[test]
    fn generic_schedule_supports_non_attention_layer_policies() {
        #[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
        enum HybridPolicy {
            Attention(AttentionPolicy),
            Recurrent,
        }

        let schedule = LayerSchedule::new(
            3,
            vec![
                HybridPolicy::Attention(AttentionPolicy::Full),
                HybridPolicy::Recurrent,
                HybridPolicy::Attention(AttentionPolicy::sliding(8).unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule.get(1), Some(&HybridPolicy::Recurrent));
        assert_eq!(schedule.get(3), None);
    }
}
