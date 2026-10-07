//! Validation of mixed categorical and dense channel payloads before allocation.

use std::collections::BTreeSet;

use eredu_core::{RealtimeChannelId, RealtimeChannelLayout, RealtimeChannelPayload};

/// Borrowed host values for one channel, in batch-major order.
#[derive(Debug, Clone, Copy)]
pub enum RealtimeChannelValues<'a> {
    /// Nonnegative categorical IDs.
    Categorical(&'a [i32]),
    /// Finite dense frame values.
    Dense(&'a [f32]),
}

/// Host-owned channel values observed after exact native completion.
/// Dense PCM/features retain their declared geometry rather than a token encoding.
#[derive(Debug, Clone, PartialEq)]
pub enum RealtimeHostChannelValues {
    /// Categorical IDs, flattened in batch-major order.
    Categorical(Vec<i32>),
    /// Dense values, flattened in batch-major order.
    Dense(Vec<f32>),
}

impl RealtimeHostChannelValues {
    /// Borrows values for shared geometry, domain and finiteness validation.
    pub fn as_values(&self) -> RealtimeChannelValues<'_> {
        match self {
            Self::Categorical(values) => RealtimeChannelValues::Categorical(values),
            Self::Dense(values) => RealtimeChannelValues::Dense(values),
        }
    }
}

/// Complete host ingress with ordinary values and selected overrides validated
/// together before any native materialization.
///
/// Identity may appear once in each group: an ordinary input and its override
/// remain distinct retained resources. Architecture policy selects required
/// channels and determines when overrides participate in execution.
pub struct ValidatedRealtimeFrameInput<'a> {
    inputs: ValidatedRealtimeChannels<'a>,
    forced: ValidatedRealtimeChannels<'a>,
    retain_diagnostics: bool,
}

impl<'a> ValidatedRealtimeFrameInput<'a> {
    /// Validates both complete payload groups without invoking a native mechanism.
    pub fn new(
        layout: &'a RealtimeChannelLayout,
        batch: usize,
        inputs: impl IntoIterator<Item = (RealtimeChannelId, RealtimeChannelValues<'a>)>,
        forced: impl IntoIterator<Item = (RealtimeChannelId, RealtimeChannelValues<'a>)>,
        retain_diagnostics: bool,
    ) -> Result<Self, RealtimeChannelIngressError> {
        let inputs = ValidatedRealtimeChannels::new(layout, batch, inputs)?;
        let forced = ValidatedRealtimeChannels::new(layout, batch, forced)?;
        Ok(Self {
            inputs,
            forced,
            retain_diagnostics,
        })
    }

    /// Exact declarations used for both payload groups.
    pub const fn channels(&self) -> &RealtimeChannelLayout {
        self.inputs.layout
    }
    /// Admitted positive batch size.
    pub const fn batch(&self) -> usize {
        self.inputs.batch
    }
    /// Whether ordered categorical diagnostics were requested.
    pub const fn retains_diagnostics(&self) -> bool {
        self.retain_diagnostics
    }

    /// Copies only previously admitted values, preserving channel identities and
    /// each payload's declared shape. A malformed override cannot partially copy
    /// ordinary inputs because both groups are validated by [`Self::new`].
    pub fn materialize<M: crate::RealtimeHostFrameMaterializer>(
        &self,
        mechanism: &mut M,
    ) -> Result<crate::MaterializedRealtimeInput<M::Tensor>, M::Error> {
        let inputs = self.inputs.materialize(mechanism)?;
        let forced = self.forced.materialize(mechanism)?;
        Ok(crate::MaterializedRealtimeInput::from_validated(
            self.inputs.layout.clone(),
            self.inputs.batch,
            inputs,
            forced,
            self.retain_diagnostics,
        ))
    }
}

pub(crate) fn validate_channel_metadata(
    layout: &RealtimeChannelLayout,
    batch: usize,
    channels: impl IntoIterator<Item = RealtimeChannelId>,
) -> Result<(), RealtimeChannelIngressError> {
    if batch == 0 {
        return Err(RealtimeChannelIngressError::EmptyBatch);
    }
    let mut seen = BTreeSet::new();
    for id in channels {
        if !seen.insert(id) {
            return Err(RealtimeChannelIngressError::Duplicate(id));
        }
        let declaration = layout
            .channel(id)
            .ok_or(RealtimeChannelIngressError::Unknown(id))?;
        if declaration.payload.elements(batch).is_none() {
            return Err(RealtimeChannelIngressError::Overflow(id));
        }
    }
    Ok(())
}

/// Validates observed current and aligned payloads before a session may publish.
/// Both collections use the same channel declarations and exact batch geometry.
/// Native observers additionally verify tensor dimensions before flattening.
pub fn validate_completed_realtime_host_frame<D>(
    frame: &crate::CompletedRealtimeFrame<RealtimeHostChannelValues, D>,
) -> Result<(), RealtimeChannelIngressError> {
    for values in [frame.values(), frame.aligned()] {
        ValidatedRealtimeChannels::new(
            frame.channels(),
            frame.batch(),
            values
                .iter()
                .map(|value| (value.channel(), value.payload().as_values())),
        )?;
    }
    Ok(())
}

/// Validated host payloads bound to their exact declared channel geometry.
///
/// Architecture policy determines which channels must be submitted for a step.
/// This mechanism validates the complete supplied set atomically, including IDs,
/// uniqueness, batch shape, categorical domains, and finite dense values.
pub struct ValidatedRealtimeChannels<'a> {
    layout: &'a RealtimeChannelLayout,
    batch: usize,
    payloads: Vec<(RealtimeChannelId, RealtimeChannelValues<'a>)>,
}

impl<'a> ValidatedRealtimeChannels<'a> {
    /// Validates every host value without invoking a native mechanism.
    pub fn new(
        layout: &'a RealtimeChannelLayout,
        batch: usize,
        payloads: impl IntoIterator<Item = (RealtimeChannelId, RealtimeChannelValues<'a>)>,
    ) -> Result<Self, RealtimeChannelIngressError> {
        let payloads = payloads.into_iter().collect::<Vec<_>>();
        validate_channel_metadata(layout, batch, payloads.iter().map(|(id, _)| *id))?;
        for (id, values) in &payloads {
            let channel = layout.channel(*id).expect("validated channel identity");
            let expected = channel
                .payload
                .elements(batch)
                .expect("validated batch geometry");
            let actual = match values {
                RealtimeChannelValues::Categorical(values) => values.len(),
                RealtimeChannelValues::Dense(values) => values.len(),
            };
            if expected != actual {
                return Err(RealtimeChannelIngressError::Shape {
                    channel: *id,
                    expected,
                    actual,
                });
            }
            match (&channel.payload, values) {
                (
                    RealtimeChannelPayload::Categorical { cardinality, .. },
                    RealtimeChannelValues::Categorical(values),
                ) => {
                    if let Some(&value) = values
                        .iter()
                        .find(|&&value| value < 0 || value as usize >= *cardinality)
                    {
                        return Err(RealtimeChannelIngressError::Category {
                            channel: *id,
                            value,
                            cardinality: *cardinality,
                        });
                    }
                }
                (RealtimeChannelPayload::Dense { .. }, RealtimeChannelValues::Dense(values)) => {
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(RealtimeChannelIngressError::NonFinite(*id));
                    }
                }
                _ => return Err(RealtimeChannelIngressError::Kind(*id)),
            }
        }
        Ok(Self {
            layout,
            batch,
            payloads,
        })
    }

    /// Exact layout that admitted these payloads.
    pub const fn layout(&self) -> &RealtimeChannelLayout {
        self.layout
    }

    /// Materializes the already validated set in caller order.
    pub fn materialize<M: crate::RealtimeHostFrameMaterializer>(
        &self,
        mechanism: &mut M,
    ) -> Result<Vec<(RealtimeChannelId, M::Tensor)>, M::Error> {
        self.payloads
            .iter()
            .map(|(id, values)| {
                let geometry = &self.layout.channel(*id).expect("validated channel").payload;
                let tensor = match (geometry, values) {
                    (
                        RealtimeChannelPayload::Categorical { width, .. },
                        RealtimeChannelValues::Categorical(values),
                    ) => mechanism.materialize_i32(values, [self.batch, *width])?,
                    (
                        RealtimeChannelPayload::Dense { shape },
                        RealtimeChannelValues::Dense(values),
                    ) => {
                        let shape = std::iter::once(self.batch)
                            .chain(shape.iter().copied())
                            .collect::<Vec<_>>();
                        mechanism.materialize_f32(values, &shape)?
                    }
                    _ => unreachable!("validated payload kind"),
                };
                Ok((*id, tensor))
            })
            .collect()
    }
}

/// Host channel input rejected before native materialization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum RealtimeChannelIngressError {
    /// Empty batches are not admitted.
    #[error("realtime channel batch is empty")]
    EmptyBatch,
    /// A channel may be submitted only once in a set.
    #[error("duplicate realtime input channel {0:?}")]
    Duplicate(RealtimeChannelId),
    /// The selected architecture did not declare the channel.
    #[error("unknown realtime input channel {0:?}")]
    Unknown(RealtimeChannelId),
    /// The batch-expanded geometry overflowed.
    #[error("realtime input channel {0:?} geometry overflowed")]
    Overflow(RealtimeChannelId),
    /// Exact element count differs.
    #[error("realtime input channel {channel:?} has {actual} values, expected {expected}")]
    Shape {
        /// Affected channel.
        channel: RealtimeChannelId,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Categorical input is outside its declared domain.
    #[error("realtime input channel {channel:?} category {value} is outside 0..{cardinality}")]
    Category {
        /// Affected channel.
        channel: RealtimeChannelId,
        /// Rejected value.
        value: i32,
        /// Exclusive domain bound.
        cardinality: usize,
    },
    /// Dense input contains a NaN or infinity.
    #[error("realtime dense input channel {0:?} contains a non-finite value")]
    NonFinite(RealtimeChannelId),
    /// Dense and categorical payloads cannot be interchanged.
    #[error("realtime input channel {0:?} payload kind differs from its declaration")]
    Kind(RealtimeChannelId),
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::{
        RealtimeHostFrameMaterializer, RealtimeHostTokenMaterializer, RealtimePayloadContract,
        RealtimePayloadGeneration, RealtimePayloadHistory, RealtimePayloadOwnerIdentity,
    };
    use eredu_core::{
        RealtimeChannelCadence, RealtimeChannelDeclaration, RealtimeChannelFeedback,
        RealtimeSlotCoordinate,
    };

    use super::*;

    fn layout() -> RealtimeChannelLayout {
        RealtimeChannelLayout::new(
            "mixed-frame-policy".into(),
            vec![
                RealtimeChannelDeclaration {
                    id: RealtimeChannelId::new(7),
                    payload: RealtimeChannelPayload::Categorical {
                        width: 2,
                        cardinality: 5,
                    },
                    cadence: RealtimeChannelCadence {
                        period: 1,
                        phase: 0,
                    },
                    delay: 0,
                    feedback: vec![],
                },
                RealtimeChannelDeclaration {
                    id: RealtimeChannelId::new(42),
                    payload: RealtimeChannelPayload::Dense { shape: vec![2, 2] },
                    cadence: RealtimeChannelCadence {
                        period: 2,
                        phase: 0,
                    },
                    delay: 2,
                    feedback: vec![RealtimeChannelFeedback {
                        source: RealtimeChannelId::new(42),
                        lag: 4,
                    }],
                },
            ],
        )
        .unwrap()
    }

    #[derive(Debug, Clone, PartialEq)]
    enum HostTensor {
        Categorical(Vec<i32>, Vec<usize>),
        Dense(Vec<f32>, Vec<usize>),
    }

    #[derive(Default)]
    struct Host {
        allocations: usize,
    }

    impl RealtimeHostTokenMaterializer for Host {
        type Tensor = HostTensor;
        type Error = Infallible;
        fn materialize_i32(
            &mut self,
            values: &[i32],
            shape: [usize; 2],
        ) -> Result<HostTensor, Infallible> {
            self.allocations += 1;
            Ok(HostTensor::Categorical(values.to_vec(), shape.to_vec()))
        }
    }

    impl RealtimeHostFrameMaterializer for Host {
        fn materialize_f32(
            &mut self,
            values: &[f32],
            shape: &[usize],
        ) -> Result<HostTensor, Infallible> {
            self.allocations += 1;
            Ok(HostTensor::Dense(values.to_vec(), shape.to_vec()))
        }
    }

    #[test]
    fn whole_frame_preserves_mixed_input_and_override_identity_without_matrix_conventions() {
        let layout = layout();
        let dense = RealtimeChannelId::new(42);
        let categorical = RealtimeChannelId::new(7);
        let prepared = ValidatedRealtimeFrameInput::new(
            &layout,
            2,
            [
                (
                    dense,
                    RealtimeChannelValues::Dense(&[0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                ),
                (
                    categorical,
                    RealtimeChannelValues::Categorical(&[1, 3, 2, 4]),
                ),
            ],
            [
                (
                    categorical,
                    RealtimeChannelValues::Categorical(&[4, 2, 3, 1]),
                ),
                (
                    dense,
                    RealtimeChannelValues::Dense(&[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, -2.0, 0.5]),
                ),
            ],
            true,
        )
        .unwrap();
        let mut mechanism = Host::default();
        let input = prepared.materialize(&mut mechanism).unwrap();
        assert_eq!(mechanism.allocations, 4);
        assert_eq!(
            input.input(categorical),
            Some(&HostTensor::Categorical(vec![1, 3, 2, 4], vec![2, 2]))
        );
        assert_eq!(
            input.forced(categorical),
            Some(&HostTensor::Categorical(vec![4, 2, 3, 1], vec![2, 2]))
        );
        assert_eq!(
            input.input(dense),
            Some(&HostTensor::Dense(
                vec![0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 2, 2]
            ))
        );
        assert_eq!(
            input.forced(dense),
            Some(&HostTensor::Dense(
                vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, -2.0, 0.5],
                vec![2, 2, 2]
            ))
        );
        assert_eq!(
            input.inputs().iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![dense, categorical]
        );
        assert_eq!(
            input
                .forced_values()
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            vec![categorical, dense]
        );
        assert!(input.retains_diagnostics());
    }

    #[test]
    fn malformed_override_rejects_whole_frame_before_copying_valid_inputs() {
        let layout = layout();
        let dense = RealtimeChannelId::new(42);
        let categorical = RealtimeChannelId::new(7);
        let mut mechanism = Host::default();
        let attempted = ValidatedRealtimeFrameInput::new(
            &layout,
            1,
            [(categorical, RealtimeChannelValues::Categorical(&[1, 2]))],
            [(
                dense,
                RealtimeChannelValues::Dense(&[1.0, 2.0, f32::INFINITY, 4.0]),
            )],
            false,
        );
        match attempted {
            Ok(input) => {
                input.materialize(&mut mechanism).unwrap();
                panic!("invalid override admitted");
            }
            Err(error) => assert_eq!(error, RealtimeChannelIngressError::NonFinite(dense)),
        }
        assert_eq!(mechanism.allocations, 0);
        assert!(
            matches!(crate::MaterializedRealtimeInput::new(layout.clone(), 1,
            vec![(categorical, ())], vec![(categorical, ()), (categorical, ())], false),
            Err(RealtimeChannelIngressError::Duplicate(id)) if id == categorical)
        );
        assert!(matches!(crate::MaterializedRealtimeInput::new(layout, 1,
            vec![], vec![(RealtimeChannelId::new(99), ())], false),
            Err(RealtimeChannelIngressError::Unknown(id)) if id == RealtimeChannelId::new(99)));
    }

    #[test]
    fn mixed_host_payloads_retain_exact_values_geometry_and_feedback_history() {
        let layout = layout();
        let payloads = [
            (
                RealtimeChannelId::new(42),
                RealtimeChannelValues::Dense(&[0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ),
            (
                RealtimeChannelId::new(7),
                RealtimeChannelValues::Categorical(&[1, 3, 2, 4]),
            ),
        ];
        let validated = ValidatedRealtimeChannels::new(&layout, 2, payloads).unwrap();
        let mut mechanism = Host::default();
        let tensors = validated.materialize(&mut mechanism).unwrap();
        assert_eq!(mechanism.allocations, 2);
        assert_eq!(
            tensors[0].1,
            HostTensor::Dense(vec![0.5, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2])
        );
        assert_eq!(
            tensors[1].1,
            HostTensor::Categorical(vec![1, 3, 2, 4], vec![2, 2])
        );
        let contract = RealtimePayloadContract::new(
            layout.clone(),
            2,
            RealtimePayloadGeneration::new(1).unwrap(),
            RealtimePayloadOwnerIdentity::new(1).unwrap(),
        )
        .unwrap();
        let mut history = RealtimePayloadHistory::with_contract(contract);
        history
            .insert_many(
                &layout,
                tensors
                    .into_iter()
                    .map(|(id, tensor)| (RealtimeSlotCoordinate::new(0, id), tensor)),
            )
            .unwrap();
        let snapshot = history.clone();
        assert_eq!(history.prune_for_next_frontier(&layout, 7).unwrap(), 0);
        // Two ticks of placement delay plus four of feedback and one completion tick.
        assert_eq!(history.prune_for_next_frontier(&layout, 8).unwrap(), 2);
        assert_eq!(snapshot.len(), 2);
        assert_eq!(
            snapshot
                .envelopes()
                .map(|e| e.geometry().clone())
                .collect::<Vec<_>>(),
            vec![
                RealtimeChannelPayload::Categorical {
                    width: 2,
                    cardinality: 5
                },
                RealtimeChannelPayload::Dense { shape: vec![2, 2] },
            ]
        );
    }

    #[test]
    fn malformed_mixed_payloads_are_rejected_before_materialization() {
        let layout = layout();
        let dense = RealtimeChannelId::new(42);
        let categorical = RealtimeChannelId::new(7);
        let mut mechanism = Host::default();
        let invalid = [
            (
                vec![(
                    dense,
                    RealtimeChannelValues::Dense(&[1.0, 2.0, f32::NAN, 4.0]),
                )],
                RealtimeChannelIngressError::NonFinite(dense),
            ),
            (
                vec![(dense, RealtimeChannelValues::Categorical(&[1, 2, 3, 4]))],
                RealtimeChannelIngressError::Kind(dense),
            ),
            (
                vec![(categorical, RealtimeChannelValues::Categorical(&[1, 5]))],
                RealtimeChannelIngressError::Category {
                    channel: categorical,
                    value: 5,
                    cardinality: 5,
                },
            ),
            (
                vec![(dense, RealtimeChannelValues::Dense(&[1.0, 2.0]))],
                RealtimeChannelIngressError::Shape {
                    channel: dense,
                    expected: 4,
                    actual: 2,
                },
            ),
            (
                vec![
                    (categorical, RealtimeChannelValues::Categorical(&[1, 2])),
                    (categorical, RealtimeChannelValues::Categorical(&[3, 4])),
                ],
                RealtimeChannelIngressError::Duplicate(categorical),
            ),
        ];
        for (payloads, expected) in invalid {
            match ValidatedRealtimeChannels::new(&layout, 1, payloads) {
                Ok(input) => {
                    input.materialize(&mut mechanism).unwrap();
                    panic!("malformed input admitted");
                }
                Err(error) => assert_eq!(error, expected),
            }
        }
        assert_eq!(mechanism.allocations, 0);
    }
}
