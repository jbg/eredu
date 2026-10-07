//! Architecture-declared categorical and dense realtime channels.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use crate::MAX_REALTIME_FRAME_DELAY;

/// Architecture-local channel identity, independent of its payload or purpose.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RealtimeChannelId(usize);

impl RealtimeChannelId {
    /// Creates an identity; a layout determines whether it is admitted.
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the architecture-local value.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Per-batch-row payload geometry. Dense channels contain finite f32 values.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RealtimeChannelPayload {
    /// Independently represented categorical values with a shared domain.
    Categorical {
        /// Number of values per batch row.
        width: usize,
        /// Exclusive upper bound of the nonnegative i32 token domain.
        cardinality: usize,
    },
    /// A row-major dense frame, excluding the leading batch dimension.
    Dense {
        /// Positive dimensions of one frame.
        shape: Vec<usize>,
    },
}

impl RealtimeChannelPayload {
    /// Checked number of scalar values for a positive batch.
    pub fn elements(&self, batch: usize) -> Option<usize> {
        if batch == 0 {
            return None;
        }
        match self {
            Self::Categorical { width, cardinality }
                if *width > 0 && *cardinality > 0 && *cardinality <= i32::MAX as usize + 1 =>
            {
                batch.checked_mul(*width)
            }
            Self::Dense { shape } if !shape.is_empty() => {
                shape
                    .iter()
                    .try_fold(batch, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
            }
            _ => None,
        }
    }
}

/// Channel activation on the architecture's explicit logical clock.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct RealtimeChannelCadence {
    /// Positive number of logical ticks between frames.
    pub period: usize,
    /// First active tick, less than the period.
    pub phase: usize,
}

impl RealtimeChannelCadence {
    /// Whether this channel is active at an explicit logical tick.
    pub fn is_active(self, tick: usize) -> bool {
        self.period > 0 && tick % self.period == self.phase
    }
}

/// A dependency on another channel's value at an explicit tick offset.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct RealtimeChannelFeedback {
    /// Source channel in the same layout.
    pub source: RealtimeChannelId,
    /// Lookback in logical ticks; zero denotes an ordered same-step dependency.
    pub lag: usize,
}

/// One categorical or dense stream and its architecture-owned timing.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct RealtimeChannelDeclaration {
    /// Stable identity within the layout.
    pub id: RealtimeChannelId,
    /// Exact per-row payload geometry and categorical domain, if applicable.
    pub payload: RealtimeChannelPayload,
    /// Activation on the explicit logical clock.
    pub cadence: RealtimeChannelCadence,
    /// Placement delay in logical ticks.
    pub delay: usize,
    /// Inputs required for this channel's computation.
    pub feedback: Vec<RealtimeChannelFeedback>,
}

/// Validated channel declarations retained as part of exact session identity.
///
/// `policy` identifies the normalized architecture coordinate policy, including
/// conventions that cannot be inferred from delay numbers alone. It is data;
/// the runtime never executes or interprets it as code.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeChannelLayout(Arc<ChannelLayoutData>);

#[derive(Debug, Eq, PartialEq, Serialize)]
struct ChannelLayoutData {
    policy: String,
    channels: Vec<RealtimeChannelDeclaration>,
}

impl Serialize for RealtimeChannelLayout {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RealtimeChannelLayout {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            policy: String,
            channels: Vec<RealtimeChannelDeclaration>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.policy, raw.channels).map_err(serde::de::Error::custom)
    }
}

impl RealtimeChannelLayout {
    /// Validates geometry, cadence, dependencies, and same-step acyclicity.
    pub fn new(
        policy: String,
        channels: Vec<RealtimeChannelDeclaration>,
    ) -> Result<Self, RealtimeChannelError> {
        if policy.is_empty() || channels.is_empty() {
            return Err(RealtimeChannelError::EmptyLayout);
        }
        let mut ids = BTreeMap::new();
        for channel in &channels {
            if ids.insert(channel.id, channel).is_some() {
                return Err(RealtimeChannelError::Duplicate(channel.id));
            }
            if channel.payload.elements(1).is_none() {
                return Err(RealtimeChannelError::Geometry(channel.id));
            }
            if channel.cadence.period == 0 || channel.cadence.phase >= channel.cadence.period {
                return Err(RealtimeChannelError::Cadence(channel.id));
            }
            if channel.delay > MAX_REALTIME_FRAME_DELAY {
                return Err(RealtimeChannelError::Delay(channel.id));
            }
        }
        let mut indegree = BTreeMap::new();
        let mut dependents: BTreeMap<RealtimeChannelId, Vec<RealtimeChannelId>> = BTreeMap::new();
        for channel in &channels {
            let mut seen = BTreeSet::new();
            let mut degree = 0usize;
            for dependency in &channel.feedback {
                if !ids.contains_key(&dependency.source)
                    || !seen.insert((dependency.source, dependency.lag))
                {
                    return Err(RealtimeChannelError::Feedback(channel.id));
                }
                if dependency.lag > MAX_REALTIME_FRAME_DELAY {
                    return Err(RealtimeChannelError::Delay(channel.id));
                }
                let source = ids[&dependency.source];
                let source_period = source.cadence.period;
                let phase = channel.cadence.phase % source_period;
                let lag = dependency.lag % source_period;
                let required_phase = if phase >= lag {
                    phase - lag
                } else {
                    source_period - (lag - phase)
                };
                if channel.cadence.period % source_period != 0
                    || required_phase != source.cadence.phase
                {
                    return Err(RealtimeChannelError::FeedbackCadence {
                        channel: channel.id,
                        producer: dependency.source,
                    });
                }
                if dependency.lag == 0 {
                    degree += 1;
                    dependents
                        .entry(dependency.source)
                        .or_default()
                        .push(channel.id);
                }
            }
            indegree.insert(channel.id, degree);
        }
        let mut ready: Vec<_> = indegree
            .iter()
            .filter_map(|(&id, &n)| (n == 0).then_some(id))
            .collect();
        let mut visited = 0usize;
        while let Some(id) = ready.pop() {
            visited += 1;
            for dependent in dependents.get(&id).into_iter().flatten() {
                let degree = indegree.get_mut(dependent).expect("validated dependency");
                *degree -= 1;
                if *degree == 0 {
                    ready.push(*dependent);
                }
            }
        }
        if visited != channels.len() {
            return Err(RealtimeChannelError::FeedbackCycle);
        }
        Ok(Self(Arc::new(ChannelLayoutData { policy, channels })))
    }

    /// Exact normalized coordinate-policy identity.
    pub fn policy(&self) -> &str {
        &self.0.policy
    }

    /// Declarations in architecture order; IDs need not be contiguous.
    pub fn channels(&self) -> &[RealtimeChannelDeclaration] {
        &self.0.channels
    }

    /// Resolves one admitted channel identity.
    pub fn channel(&self, id: RealtimeChannelId) -> Option<&RealtimeChannelDeclaration> {
        self.0.channels.iter().find(|channel| channel.id == id)
    }

    /// Conservative tick lookback covering both delays and feedback.
    pub fn retention_lookback(&self) -> usize {
        let delay = self.0.channels.iter().map(|c| c.delay).max().unwrap_or(0);
        let feedback = self
            .0
            .channels
            .iter()
            .flat_map(|c| &c.feedback)
            .map(|d| d.lag)
            .max()
            .unwrap_or(0);
        // Both terms are individually bounded to i32::MAX, including on 32-bit hosts.
        delay + feedback
    }
}

/// Invalid architecture channel declaration, detected without native resources.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum RealtimeChannelError {
    /// Both policy identity and channels must be present.
    #[error("realtime channel layout or policy is empty")]
    EmptyLayout,
    /// IDs must be unique.
    #[error("duplicate realtime channel {0:?}")]
    Duplicate(RealtimeChannelId),
    /// Empty, unrepresentable, or overflowing payload geometry.
    #[error("invalid realtime channel geometry for {0:?}")]
    Geometry(RealtimeChannelId),
    /// Invalid period or phase.
    #[error("invalid realtime channel cadence for {0:?}")]
    Cadence(RealtimeChannelId),
    /// Delay or lag exceeds portable coordinates.
    #[error("realtime channel delay exceeds portable coordinates for {0:?}")]
    Delay(RealtimeChannelId),
    /// A dependency is missing or repeated.
    #[error("invalid realtime feedback dependency for {0:?}")]
    Feedback(RealtimeChannelId),
    /// The source is inactive on a tick required by the consumer.
    #[error("realtime channel {channel:?} requires inactive source {producer:?}")]
    FeedbackCadence {
        /// Dependent channel.
        channel: RealtimeChannelId,
        /// Source channel with incompatible cadence.
        producer: RealtimeChannelId,
    },
    /// Same-step dependencies cannot form a cycle; delayed feedback may.
    #[error("realtime same-step feedback contains a cycle")]
    FeedbackCycle,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(id: usize, source: usize, lag: usize) -> RealtimeChannelDeclaration {
        RealtimeChannelDeclaration {
            id: RealtimeChannelId::new(id),
            payload: RealtimeChannelPayload::Dense { shape: vec![2, 3] },
            cadence: RealtimeChannelCadence {
                period: 2,
                phase: 1,
            },
            delay: 2,
            feedback: vec![RealtimeChannelFeedback {
                source: RealtimeChannelId::new(source),
                lag,
            }],
        }
    }

    #[test]
    fn delayed_feedback_is_legal_but_same_step_cycles_are_not() {
        let channels = vec![channel(9, 3, 2), channel(3, 9, 0)];
        let layout = RealtimeChannelLayout::new("coupled".into(), channels.clone()).unwrap();
        assert_eq!(layout.retention_lookback(), 4);
        assert_eq!(
            layout
                .channel(RealtimeChannelId::new(9))
                .unwrap()
                .payload
                .elements(2),
            Some(12)
        );
        let cadence = channels[0].cadence;
        assert!(!cadence.is_active(0));
        assert!(cadence.is_active(1));
        assert!(!cadence.is_active(2));
        assert!(cadence.is_active(3));

        let mut invalid = channels.clone();
        invalid[0].feedback[0].lag = 0;
        assert_eq!(
            RealtimeChannelLayout::new("coupled".into(), invalid),
            Err(RealtimeChannelError::FeedbackCycle)
        );
        let mut invalid = channels.clone();
        invalid[0].feedback[0].source = RealtimeChannelId::new(99);
        assert_eq!(
            RealtimeChannelLayout::new("coupled".into(), invalid),
            Err(RealtimeChannelError::Feedback(RealtimeChannelId::new(9)))
        );
        let mut invalid = channels;
        invalid[0].feedback[0].lag = 1;
        assert!(matches!(
            RealtimeChannelLayout::new("coupled".into(), invalid),
            Err(RealtimeChannelError::FeedbackCadence { .. })
        ));
    }

    #[test]
    fn serialized_layout_cannot_bypass_geometry_or_dependency_admission() {
        let layout = RealtimeChannelLayout::new("coupled".into(), vec![channel(9, 9, 2)]).unwrap();
        let value = serde_json::to_value(&layout).unwrap();
        assert_eq!(
            serde_json::from_value::<RealtimeChannelLayout>(value.clone()).unwrap(),
            layout
        );
        for geometry in [vec![0], vec![], vec![usize::MAX, 2]] {
            let mut malformed = value.clone();
            malformed["channels"][0]["payload"]["shape"] = serde_json::json!(geometry);
            assert!(serde_json::from_value::<RealtimeChannelLayout>(malformed).is_err());
        }
        let mut malformed = value.clone();
        malformed["channels"][0]["feedback"][0]["lag"] = 0.into();
        assert!(serde_json::from_value::<RealtimeChannelLayout>(malformed).is_err());
        let mut malformed = value;
        malformed["channels"][0]["cadence"]["period"] = 0.into();
        assert!(serde_json::from_value::<RealtimeChannelLayout>(malformed).is_err());
    }
}
