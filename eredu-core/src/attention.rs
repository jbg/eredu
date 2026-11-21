//! Architecture-neutral decoder layer schedules and attention geometry.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, num::NonZeroU32};

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

    /// Converts an optional signed runtime window into an exact attention policy.
    pub fn from_sliding_window(window: Option<i32>) -> Result<Self, LayerScheduleError> {
        match window {
            None => Ok(Self::Full),
            Some(window) if window <= 0 => Err(LayerScheduleError::ZeroWindow),
            Some(window) => Self::sliding(window as u32),
        }
    }

    /// Returns the exact signed runtime window, rejecting values outside `i32`.
    pub fn sliding_window_i32(self) -> Result<Option<i32>, LayerScheduleError> {
        self.window()
            .map(|window| {
                i32::try_from(window.get()).map_err(|_| LayerScheduleError::WindowOutOfRange {
                    window: window.get(),
                })
            })
            .transpose()
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
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerSchedule<P> {
    layers: Box<[P]>,
}

impl<P> LayerSchedule<P> {
    /// Represents a partition owning no local invocations. Whole-family
    /// construction through `new` still requires a positive layer count.
    pub fn empty() -> Self {
        Self {
            layers: Box::new([]),
        }
    }

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
    /// Creates an all-sliding attention schedule.
    pub fn all_sliding(layer_count: usize, window: u32) -> Result<Self, LayerScheduleError> {
        Self::new(
            layer_count,
            vec![AttentionPolicy::sliding(window)?; layer_count],
        )
    }
    /// Creates a schedule from a Boolean pattern where `true` means sliding.
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
        let policy = match (pattern.iter().any(|value| *value), window) {
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
                        policy.expect("validated")
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
            .filter(|p| matches!(p, AttentionPolicy::Full))
            .count()
    }
    /// Returns the number of sliding-attention layers.
    pub fn sliding_layer_count(&self) -> usize {
        self.len() - self.full_layer_count()
    }
    /// Counts sliding layers by exact window.
    pub fn sliding_windows(&self) -> BTreeMap<NonZeroU32, usize> {
        let mut result = BTreeMap::new();
        for window in self.iter().copied().filter_map(AttentionPolicy::window) {
            *result.entry(window).or_default() += 1;
        }
        result
    }
    /// Returns a stable representation suitable for fingerprints.
    pub fn fingerprint_component(&self) -> String {
        self.iter()
            .map(|policy| match policy {
                AttentionPolicy::Full => "f".into(),
                AttentionPolicy::Sliding { window } => format!("s{}", window.get()),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Validation error for an ordered per-layer schedule.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum LayerScheduleError {
    /// A decoder-layer schedule cannot be empty.
    #[error("layer schedule must contain at least one layer")]
    Empty,
    /// The supplied policy count differs from decoder depth.
    #[error("layer schedule has {actual} entries for {expected} decoder layers")]
    LayerCount {
        /// Decoder layer count.
        expected: usize,
        /// Supplied policy count.
        actual: usize,
    },
    /// A sliding window was zero.
    #[error("sliding attention window must be positive")]
    ZeroWindow,
    /// Sliding attention was enabled without a window.
    #[error("sliding attention is enabled for at least one layer without a window")]
    MissingWindow,
    /// A window cannot be represented by runtime cache APIs.
    #[error("sliding attention window {window} exceeds i32")]
    WindowOutOfRange {
        /// Unrepresentable window.
        window: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_attention_schedule() {
        let schedule = LayerSchedule::new(
            3,
            vec![
                AttentionPolicy::Full,
                AttentionPolicy::sliding(8).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        assert_eq!(schedule.fingerprint_component(), "f,s8,f");
        assert_eq!(schedule.sliding_layer_count(), 1);
        assert!(LayerSchedule::from_sliding_pattern(2, &[true], Some(4)).is_err());
    }
}

/// Side-effect-free native facts for tiled explicit input-score attention.
/// Copies and score bytes are conservative live-buffer allowances, not payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputScoreAttentionMechanism {
    /// Query-by-key elements per head before query tiling.
    pub score_tile_elements: u64,
    /// Maximum query rows in one tile.
    pub max_query_rows: u64,
    /// Fallback expanded K/V and contiguous projection copies per query tile,
    /// used when `full_key_tiles` is absent or does not cover the key length.
    pub key_value_copies: u64,
    /// Working bytes per score element, including precision conversions.
    pub score_bytes: u64,
    /// Optional full-key layout reuse and bounded tile-graph retention. Missing
    /// facts retain the legacy per-tile copy allowance for all key lengths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_key_tiles: Option<FullKeyAttentionTiles>,
}

/// Native retention contract for tiles that each attend to the entire key row.
/// Beyond this contract's key limit, the mechanism's legacy allowances apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullKeyAttentionTiles {
    /// Largest key row covered by shared preparation and bounded evaluation.
    pub max_key_positions: u64,
    /// Expanded K/V and native contiguous layouts, in query-width payloads.
    /// This is charged once per invocation rather than once per query tile.
    pub shared_key_value_copies: u64,
    /// Largest number of unevaluated query tiles retained simultaneously.
    pub max_live_query_tiles: u64,
    /// Whole-query output payloads retained, including final concatenation.
    pub retained_output_copies: u64,
    /// When the query tile count exceeds `max_live_query_tiles`, each batch,
    /// including the last, completes synchronously and detaches evaluated input
    /// dependency graphs. This releases unowned intermediates, not owner-held
    /// residuals, state views, completed outputs, parameters or allocator cache.
    /// Absent legacy facts do not establish an upstream release boundary.
    #[serde(default)]
    pub evaluates_input_dependencies: bool,
}
