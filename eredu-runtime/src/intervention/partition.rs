//! Global activation intent projected onto a prepaid local invocation.
//!
//! This is the local mechanism used by distributed composition. It preserves the
//! original admission; enclosing execution must still bind invocation ownership,
//! agree all-rank intent, and publish outcomes only after shared completion.
use super::*;
use eredu_core::component::ComponentCoordinateMap;
use sha2::{Digest, Sha256};

/// One actual invocation member, including nonexporting replicas and empty shards.
pub struct PartitionActivationMember<'a> {
    /// World rank that executes this invocation.
    pub rank: usize,
    /// Original-plan geometry for this member's actual tensor.
    pub projection: PartitionActivationProjection<'a>,
}

/// Retained sparse placement for every executing rank, including replicas.
pub struct PartitionRoutedActivationMember<'a> {
    /// World rank of the invocation.
    pub rank: usize,
    /// Exact expert/unit coordinates and exchange topology. Publication peer
    /// selects outcome counting only; the operation affects all received peers.
    pub ownership: &'a RoutedUnitCaptureOwnership,
    /// Actual provider input width before native route chunking.
    pub input_width: u64,
}

/// Architecture-owned placement; runtime and native completion remain separate.
pub trait PartitionActivationLayout {
    /// Projects independently shaped forwards through the retained invocation owners.
    fn activation_members_at<'a>(
        &'a self,
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        max_members: usize,
        max_regions: usize,
    ) -> Result<Vec<PartitionActivationMember<'a>>, CaptureError> {
        if invocation.is_some() {
            return Err(CaptureError::Unsupported(
                "partition layout has no explicit intervention invocation projection".into(),
            ));
        }
        self.activation_members(plan, operation, phase, prediction, max_members, max_regions)
    }
    /// Sparse invocation membership, borrowed from retained architecture placement.
    /// The caller prepays bounded metadata before this method creates its list.
    fn routed_activation_members<'a>(
        &'a self,
        _plan: &AdmittedInterventionPlan,
        _operation: usize,
        _max_members: usize,
    ) -> Result<Vec<PartitionRoutedActivationMember<'a>>, CaptureError> {
        Err(CaptureError::Unsupported(
            "sparse intervention placement unavailable".into(),
        ))
    }
    /// Cold upper bound on region metadata created during projection. The default
    /// uses the caller's full bound; retained layouts may tighten it using their
    /// actual partition axes. No payload or native work may occur here.
    fn activation_region_bound(
        &self,
        _plan: &AdmittedInterventionPlan,
        _operation: usize,
        _max_members: usize,
        max_regions: usize,
    ) -> Result<usize, CaptureError> {
        Ok(max_regions)
    }

    /// Projects every actual invocation member in ascending rank order. Limits
    /// bound both membership and the total number of native update regions.
    fn activation_members<'a>(
        &'a self,
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        max_members: usize,
        max_regions: usize,
    ) -> Result<Vec<PartitionActivationMember<'a>>, CaptureError>;
}

struct Region {
    local: ResolvedCaptureSlice,
    destination: Option<ResolvedCaptureSlice>,
}

/// Exact local geometry retaining the original immutable global operation.
/// Construction copies no intervention payload and performs no native work.
/// Fragment storage is bounded by the caller's explicit limit.
pub struct PartitionActivationProjection<'a> {
    plan: &'a AdmittedInterventionPlan,
    operation: usize,
    phase: CapturePhase,
    prediction: u64,
    shape: Vec<u64>,
    coordinates: &'a ComponentCoordinateMap,
    local_columns: bool,
    regions: Vec<Region>,
    sum_offset_owner: Option<bool>,
}

impl<'a> PartitionActivationProjection<'a> {
    /// Projects a scheduled operation through architecture-owned coordinates.
    /// Replicas require their own projection; export ownership is irrelevant.
    pub fn new(
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        global_shape: &[u64],
        axis: usize,
        coordinates: &'a ComponentCoordinateMap,
        max_fragments: usize,
    ) -> Result<Self, CaptureError> {
        Self::new_at(
            plan,
            operation,
            phase,
            prediction,
            None,
            global_shape,
            axis,
            coordinates,
            max_fragments,
        )
    }

    /// Projects one actual invocation while retaining its original geometry authority.
    pub fn new_at(
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        global_shape: &[u64],
        axis: usize,
        coordinates: &'a ComponentCoordinateMap,
        max_fragments: usize,
    ) -> Result<Self, CaptureError> {
        let action = &plan
            .plan()
            .operations
            .get(operation)
            .ok_or_else(|| invalid("unknown partition intervention operation"))?
            .action;
        let dtype = action.dtype().ok_or_else(|| {
            CaptureError::Unsupported(
                "routing control requires a routing-specific partition contract".into(),
            )
        })?;
        let slice = plan.validate_at(
            operation,
            phase,
            prediction,
            invocation,
            global_shape,
            Some(dtype),
        )?;
        if max_fragments == 0
            || axis >= global_shape.len()
            || global_shape[axis] != coordinates.global_count() as u64
        {
            return Err(invalid(
                "partition intervention axis or fragment bound is invalid",
            ));
        }
        let column_mask = matches!(
            action,
            InterventionAction::MaskComponents { .. } | InterventionAction::MaskLogits { .. }
        );
        let last = global_shape.len() - 1;
        if column_mask
            && (slice.starts[last] != 0
                || slice.ends[last] != global_shape[last]
                || slice.strides[last] != 1)
        {
            return Err(invalid(
                "column intervention requires the complete global final axis",
            ));
        }
        let mut shape = global_shape.to_vec();
        shape[axis] = coordinates.local_count() as u64;
        let local_columns = column_mask && axis == last;
        let regions = if local_columns {
            // A compact mask remains one local operation even for a permuted
            // axis. Expanding it to one fragment per component defeats its bound.
            if coordinates.local_count() == 0 {
                vec![]
            } else {
                let mut local = slice;
                local.ends[last] = shape[last];
                local.shape[last] = shape[last];
                vec![Region {
                    local,
                    destination: None,
                }]
            }
        } else {
            CaptureSlicePartition::new(global_shape, &slice, axis, coordinates, max_fragments)?
                .fragments()
                .iter()
                .map(|fragment| Region {
                    local: fragment.local().clone(),
                    destination: Some(fragment.destination().clone()),
                })
                .collect()
        };
        Ok(Self {
            plan,
            operation,
            phase,
            prediction,
            shape,
            coordinates,
            local_columns,
            regions,
            sum_offset_owner: None,
        })
    }

    /// Treats this actual tensor as one additive term of the complete activation.
    /// The architecture must designate exactly one offset owner in each sum.
    /// Zero, scaling and masks affect every term; Add affects only the owner,
    /// while Replace clears the other terms. Native floating-point rounding still
    /// occurs on each term, so equality with editing a reduced tensor is numerical,
    /// not bitwise. This does not add a native reduction or completion authority.
    pub fn as_sum_term(mut self, offset_owner: bool) -> Result<Self, CaptureError> {
        if self.sum_offset_owner.is_some()
            || self.coordinates.local_count() != self.coordinates.global_count()
            || (0..self.coordinates.local_count())
                .any(|index| self.coordinates.local_to_global(index) != Some(index))
        {
            return Err(invalid("an additive intervention term requires complete ordered coordinates and one projection"));
        }
        let action = &self.plan.plan().operations[self.operation].action;
        if matches!(action, InterventionAction::MaskLogits { .. }) {
            return Err(CaptureError::Unsupported(
                "vocabulary sentinel masking is not an additive activation mask".into(),
            ));
        }
        self.sum_offset_owner = Some(offset_owner);
        if !offset_owner && matches!(action, InterventionAction::Add { .. }) {
            // This rank still validates its source and participates in completion,
            // but creates no payload copy or native update for another rank's offset.
            self.regions.clear();
        }
        Ok(self)
    }

    /// Actual local tensor geometry, including empty shards.
    pub fn local_shape(&self) -> &[u64] {
        &self.shape
    }

    /// Number of disjoint native updates. Zero denotes no local overlap.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Stable exact geometry identity for common pre-forward agreement. Includes
    /// ordered coordinates and local/global region correspondence, so equally
    /// sized permutations cannot substitute for the retained placement.
    pub fn geometry_identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"eredu-partition-activation-geometry-v2\0");
        digest.update(self.plan.intent_identity().as_bytes());
        digest.update((self.operation as u64).to_le_bytes());
        digest.update([u8::from(self.local_columns)]);
        digest.update([match self.sum_offset_owner {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }]);
        let vector = |digest: &mut Sha256, values: &[u64]| {
            digest.update((values.len() as u64).to_le_bytes());
            for value in values {
                digest.update(value.to_le_bytes());
            }
        };
        vector(&mut digest, &self.shape);
        digest.update((self.coordinates.global_count() as u64).to_le_bytes());
        digest.update((self.coordinates.local_count() as u64).to_le_bytes());
        for local in 0..self.coordinates.local_count() {
            digest.update(
                (self
                    .coordinates
                    .local_to_global(local)
                    .expect("retained coordinate") as u64)
                    .to_le_bytes(),
            );
        }
        digest.update((self.regions.len() as u64).to_le_bytes());
        for region in &self.regions {
            digest.update([u8::from(region.destination.is_some())]);
            for slice in std::iter::once(&region.local).chain(region.destination.iter()) {
                for values in [&slice.starts, &slice.ends, &slice.strides, &slice.shape] {
                    vector(&mut digest, values);
                }
            }
        }
        digest.finalize().into()
    }

    /// Charges host projection before copying payloads, then charges the complete
    /// native estimate before granting move-only local work. Failure or dropping
    /// the returned work never refunds either charge. No backend method is called.
    pub fn reserve(
        self,
        reservation: &mut impl CaptureReservation,
        estimator: &dyn InterventionEstimator,
    ) -> Result<ReservedPartitionActivation, CaptureError> {
        let original_action = &self.plan.plan().operations[self.operation].action;
        let cleared = InterventionAction::Zero {
            dtype: original_action.dtype().expect("activation"),
        };
        let action = if self.sum_offset_owner == Some(false)
            && matches!(original_action, InterventionAction::Replace { .. })
        {
            &cleared
        } else {
            original_action
        };
        let mut host_bytes = add(
            256,
            add(
                self.plan.identity().len() as u64,
                add(
                    self.plan.intent_identity().len() as u64,
                    mul(self.shape.len() as u64, 8)?,
                )?,
            )?,
        )?;
        for region in &self.regions {
            estimator.validate_geometry(&self.shape, &region.local)?;
            let payload = match action {
                InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => mul(
                    elements(&region.local.shape)?,
                    if tensor.values.dtype() == InterventionDtype::Float32 {
                        4
                    } else {
                        2
                    },
                )?,
                InterventionAction::Mask { .. } => elements(&region.local.shape)?,
                // Local index storage plus bounded uniqueness-validation scratch.
                InterventionAction::MaskComponents { indices, .. } => {
                    mul(indices.len() as u64, 96)?
                }
                InterventionAction::MaskLogits { token_ids, .. } => {
                    mul(token_ids.len() as u64, 96)?
                }
                _ => 0,
            };
            // Owned action/region envelopes and exact row-major payload storage.
            host_bytes = add(
                host_bytes,
                add(256, add(mul(self.shape.len() as u64, 48)?, payload)?)?,
            )?;
        }
        let host = CaptureUsage {
            host_bytes,
            ..Default::default()
        };
        let mut projected = reservation.reserve_quota(host)?;
        reserve_envelope(&mut projected, host)?;
        let mut updates = Vec::with_capacity(self.regions.len());
        let mut native = CaptureUsage::default();
        for region in self.regions {
            let local_action = if self.local_columns {
                match action {
                    InterventionAction::MaskComponents {
                        dtype,
                        indices,
                        keep_selected,
                    } => InterventionAction::MaskComponents {
                        dtype: *dtype,
                        indices: self
                            .coordinates
                            .localize_indices(indices)
                            .map_err(|e| invalid(&e.to_string()))?,
                        keep_selected: *keep_selected,
                    },
                    InterventionAction::MaskLogits { dtype, token_ids } => {
                        InterventionAction::MaskLogits {
                            dtype: *dtype,
                            token_ids: self
                                .coordinates
                                .localize_indices(token_ids)
                                .map_err(|e| invalid(&e.to_string()))?,
                        }
                    }
                    _ => return Err(invalid("invalid local column action")),
                }
            } else {
                project_action(
                    action,
                    region.destination.as_ref().expect("fragment destination"),
                )?
            };
            if matches!(&local_action, InterventionAction::MaskLogits { token_ids, .. } if token_ids.is_empty())
            {
                // Every excluded global token belongs to another shard. There
                // is no native update here, but the invocation still participates.
                continue;
            }
            validate_local_action(
                &local_action,
                local_action.dtype().expect("activation"),
                &region.local.shape,
            )?;
            let usage = estimator.activation_usage(&self.shape, &region.local, &local_action)?;
            if usage.captures != 0 || usage.encoded_bytes != 0 {
                return Err(invalid(
                    "activation estimate includes unrelated capture or record work",
                ));
            }
            native = native.checked_add(usage)?;
            updates.push((region.local, local_action, usage));
        }
        let quota = reservation.reserve_quota(native)?;
        Ok(ReservedPartitionActivation {
            plan_identity: self.plan.identity().into(),
            intent_identity: self.plan.intent_identity().into(),
            dtype: action.dtype().expect("activation"),
            operation: self.operation,
            phase: self.phase,
            prediction: self.prediction,
            shape: self.shape,
            updates,
            quota,
            charged: host.checked_add(native)?,
        })
    }
}

/// Move-only prepaid local activation work. This cannot be deserialized, cloned,
/// or rebound to a different plan. Enclosing distributed execution supplies the
/// invocation/epoch authority and failure agreement; this grants no such authority.
pub struct ReservedPartitionActivation {
    plan_identity: String,
    intent_identity: String,
    dtype: InterventionDtype,
    operation: usize,
    phase: CapturePhase,
    prediction: u64,
    shape: Vec<u64>,
    updates: Vec<(ResolvedCaptureSlice, InterventionAction, CaptureUsage)>,
    quota: CaptureQuota,
    charged: CaptureUsage,
}

impl ReservedPartitionActivation {
    /// Exact original source/session/semantics admission.
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }
    /// Global intent for peer agreement, separate from local session authority.
    pub fn intent_identity(&self) -> &str {
        &self.intent_identity
    }
    /// Operation ordinal in the original plan, preserving composition order.
    pub fn operation_index(&self) -> usize {
        self.operation
    }
    /// Actual phase retained at preparation.
    pub fn phase(&self) -> CapturePhase {
        self.phase
    }
    /// Prediction ordinal, independent of token-position slices.
    pub fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Logical host/native charges already consumed by the parent reservation.
    pub fn charged(&self) -> CaptureUsage {
        self.charged
    }
    /// Whether this invocation needs no native update (no overlap, or another
    /// additive term owns the entire Add offset).
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    /// Checks the exact retained local shape and dtype without submitting work.
    pub fn validate_source<B: InterventionBackend>(
        &self,
        backend: &B,
        input: &B::Tensor,
    ) -> Result<(), CaptureExecutionError<B::Error>> {
        let shape = backend
            .shape(input)
            .map_err(CaptureExecutionError::Backend)?;
        let dtype = backend
            .intervention_dtype(input)
            .map_err(CaptureExecutionError::Backend)?;
        if shape != self.shape || self.dtype != dtype {
            return Err(invalid(
                "partition intervention source geometry or dtype changed after preparation",
            )
            .into());
        }
        Ok(())
    }

    /// Executes each disjoint projected update with the ordinary activation
    /// mechanism, consuming this work exactly once. A no-overlap result is an
    /// acknowledgment (`None`), never a manufactured zero tensor. No completion
    /// or global outcome is implied by successfully submitting these operations.
    pub fn apply<B: InterventionBackend>(
        mut self,
        backend: &mut B,
        input: &B::Tensor,
    ) -> Result<Option<B::Tensor>, CaptureExecutionError<B::Error>> {
        self.validate_source(backend, input)?;
        let mut effective = None;
        for (slice, action, usage) in self.updates {
            reserve_envelope(&mut self.quota, usage)?;
            let value = effective.as_ref().unwrap_or(input);
            effective = Some(super::activation::apply_partition_activation(
                backend, value, &action, &slice,
            )?);
        }
        Ok(effective)
    }
}

fn project_action(
    action: &InterventionAction,
    destination: &ResolvedCaptureSlice,
) -> Result<InterventionAction, CaptureError> {
    Ok(match action {
        InterventionAction::Mask { dtype, shape, keep } => InterventionAction::Mask {
            dtype: *dtype,
            shape: destination.shape.clone(),
            keep: select_payload(keep, shape, destination)?,
        },
        InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
            let values = match &tensor.values {
                InterventionValues::Float32(values) => {
                    InterventionValues::Float32(select_payload(values, &tensor.shape, destination)?)
                }
                InterventionValues::Float16(values) => {
                    InterventionValues::Float16(select_payload(values, &tensor.shape, destination)?)
                }
                InterventionValues::Bfloat16(values) => InterventionValues::Bfloat16(
                    select_payload(values, &tensor.shape, destination)?,
                ),
            };
            let tensor = InterventionTensor {
                shape: destination.shape.clone(),
                values,
            };
            if matches!(action, InterventionAction::Replace { .. }) {
                InterventionAction::Replace { tensor }
            } else {
                InterventionAction::Add { tensor }
            }
        }
        action => action.clone(),
    })
}

// Iterate only selected elements; no global-size index vector or F16/BF16 cast.
fn select_payload<T: Copy>(
    values: &[T],
    shape: &[u64],
    slice: &ResolvedCaptureSlice,
) -> Result<Vec<T>, CaptureError> {
    let count = elements(&slice.shape)?;
    let mut output =
        Vec::with_capacity(usize::try_from(count).map_err(|_| CaptureError::Overflow)?);
    for ordinal in 0..count {
        let mut remaining = ordinal;
        let mut source = 0u64;
        let mut stride = 1u64;
        for axis in (0..shape.len()).rev() {
            let coordinate = add(
                slice.starts[axis],
                mul(remaining % slice.shape[axis], slice.strides[axis])?,
            )?;
            remaining /= slice.shape[axis];
            source = add(source, mul(coordinate, stride)?)?;
            stride = mul(stride, shape[axis])?;
        }
        output.push(
            *values
                .get(usize::try_from(source).map_err(|_| CaptureError::Overflow)?)
                .ok_or_else(|| {
                    invalid("partition intervention payload destination is outside admission")
                })?,
        );
    }
    Ok(output)
}

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

// Global admission prohibits masking the entire model vocabulary. A valid
// global mask may cover every entry of one local shard; only this private,
// admission-preserving path permits it. Ordinary direct calls stay unchanged.
pub(super) fn validate_local_action(
    action: &InterventionAction,
    dtype: InterventionDtype,
    shape: &[u64],
) -> Result<(), CaptureError> {
    if let InterventionAction::MaskLogits {
        dtype: declared,
        token_ids,
    } = action
    {
        if shape.last().copied() == Some(token_ids.len() as u64) {
            InterventionAction::Zero { dtype: *declared }
                .validate_activation_region(dtype, shape)?;
            let mut unique = std::collections::BTreeSet::new();
            if token_ids
                .iter()
                .any(|id| u64::from(*id) >= token_ids.len() as u64 || !unique.insert(*id))
            {
                return Err(invalid("invalid projected vocabulary mask"));
            }
            return Ok(());
        }
    }
    action.validate_activation_region(dtype, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_projection_preserves_float16_and_bfloat16_bits_and_stride_origins() {
        let destination = ResolvedCaptureSlice {
            starts: vec![0, 1],
            ends: vec![2, 4],
            strides: vec![1, 2],
            shape: vec![2, 2],
        };
        for values in [
            InterventionValues::Float16(vec![
                0x8000, 0x3c01, 0x0001, 0x7bff, 0xbc00, 0x3555, 0x0400, 0x3bff,
            ]),
            InterventionValues::Bfloat16(vec![
                0x8000, 0x3f81, 0x0001, 0x7f7f, 0xbf80, 0x3eab, 0x0080, 0x3f7f,
            ]),
        ] {
            let original = InterventionTensor {
                shape: vec![2, 4],
                values,
            };
            for action in [
                InterventionAction::Replace {
                    tensor: original.clone(),
                },
                InterventionAction::Add {
                    tensor: original.clone(),
                },
            ] {
                let projected = project_action(&action, &destination).unwrap();
                let tensor = match &projected {
                    InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
                        tensor
                    }
                    _ => panic!(),
                };
                assert_eq!(tensor.shape, [2, 2]);
                assert_eq!(projected.kind(), action.kind());
                let (InterventionValues::Float16(before), InterventionValues::Float16(after)) =
                    (&original.values, &tensor.values)
                else {
                    let (InterventionValues::Bfloat16(before), InterventionValues::Bfloat16(after)) =
                        (&original.values, &tensor.values)
                    else {
                        panic!()
                    };
                    assert_eq!(after, &[before[1], before[3], before[5], before[7]]);
                    continue;
                };
                assert_eq!(after, &[before[1], before[3], before[5], before[7]]);
            }
        }
    }
}
