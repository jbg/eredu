//! Exact tensor bundles retained and published inside the ordinary forward transaction.
use super::*;
use sha2::{Digest, Sha256};

/// One architecture-selected producer and exact invocation tensor geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardCaptureSlot {
    role: String,
    owner: usize,
    shape: Vec<i32>,
    placeholder_element: eredu_nn::TensorElementType,
    permitted_elements: Vec<eredu_nn::TensorElementType>,
}
impl ForwardCaptureSlot {
    /// Selects a complete tensor after any architecture-required reductions.
    pub fn new(
        role: String,
        owner: usize,
        shape: Vec<i32>,
        element: eredu_nn::TensorElementType,
    ) -> Result<Self, String> {
        if role.trim().is_empty() || shape.is_empty() || shape.iter().any(|n| *n <= 0) {
            return Err("capture slot has an empty identity or invalid shape".into());
        }
        shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
            .ok_or("capture slot element count overflows")?;
        Ok(Self {
            role,
            owner,
            shape,
            placeholder_element: element,
            permitted_elements: vec![element],
        })
    }
    /// Retains the floating precision envelope admitted for this producer.
    /// Publication preserves the owner's actual type through shared metadata transport.
    pub fn with_permitted_elements(
        mut self,
        mut elements: Vec<eredu_nn::TensorElementType>,
    ) -> Result<Self, String> {
        use eredu_nn::TensorElementType as E;
        let order = |element: &E| match element {
            E::F16 => 1,
            E::Bf16 => 2,
            E::F32 => 3,
            E::F64 => 4,
            _ => 0,
        };
        if elements.is_empty()
            || elements.iter().any(|e| order(e) == 0)
            || !elements.contains(&self.placeholder_element)
        {
            return Err("capture precision envelope is invalid or omits its placeholder".into());
        }
        elements.sort_by_key(order);
        elements.dedup();
        self.permitted_elements = elements;
        Ok(self)
    }

    /// Stable ordered capture identity.
    pub fn role(&self) -> &str {
        &self.role
    }
    /// Exact world rank supplying the complete tensor.
    pub const fn owner(&self) -> usize {
        self.owner
    }
    /// Resolved invocation dimensions, shared by source and destination storage.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Placeholder precision before the owner's actual type is published.
    pub const fn placeholder_element(&self) -> eredu_nn::TensorElementType {
        self.placeholder_element
    }
    /// Allowed native producer types; source and destination retain the same actual type.
    pub fn permitted_elements(&self) -> &[eredu_nn::TensorElementType] {
        &self.permitted_elements
    }
}

/// An architecture-authored ordered publication contract for one forward invocation.
/// This contains no tensor, mutable state, device or communication authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardCaptureBundle {
    slots: Vec<ForwardCaptureSlot>,
    descriptor: [u64; 8],
}
impl ForwardCaptureBundle {
    /// Retains complete producer/geometry identities before native submission.
    pub fn new(slots: Vec<ForwardCaptureSlot>) -> Result<Self, String> {
        if slots.is_empty()
            || slots.iter().map(|s| &s.role).collect::<BTreeSet<_>>().len() != slots.len()
        {
            return Err("capture bundle identities are empty or repeated".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"eredu-forward-capture-bundle-v2\0");
        hash.update((slots.len() as u64).to_le_bytes());
        for slot in &slots {
            hash.update((slot.role.len() as u64).to_le_bytes());
            hash.update(slot.role.as_bytes());
            hash.update((slot.owner as u64).to_le_bytes());
            hash.update((slot.shape.len() as u64).to_le_bytes());
            for dimension in &slot.shape {
                hash.update(dimension.to_le_bytes());
            }
            // The enum's explicit semantic name participates, not its Rust layout.
            let precision = format!(
                "{:?}/{:?}",
                slot.placeholder_element, slot.permitted_elements
            );
            hash.update((precision.len() as u64).to_le_bytes());
            hash.update(precision.as_bytes());
        }
        let digest = hash.finalize();
        let mut descriptor = [0u64; 8];
        descriptor[0] = 0x4552_4342;
        descriptor[1] = slots.len() as u64;
        for (word, bytes) in descriptor[2..6].iter_mut().zip(digest.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().unwrap());
        }
        Ok(Self { slots, descriptor })
    }
    /// Exact canonical transfer order.
    pub fn slots(&self) -> &[ForwardCaptureSlot] {
        &self.slots
    }
    fn validate<T: Tensor>(&self, values: &[T]) -> Result<(), String> {
        if values.len() != self.slots.len() {
            return Err("capture bundle tensor count differs from its contract".into());
        }
        for (slot, value) in self.slots.iter().zip(values) {
            if value.shape() != slot.shape
                || !value
                    .element_type()
                    .is_some_and(|element| slot.permitted_elements.contains(&element))
            {
                return Err(format!(
                    "capture tensor geometry or dtype differs for {}: expected {:?} with {:?}, got {:?} with {:?}",
                    slot.role, slot.shape, slot.permitted_elements, value.shape(), value.element_type()
                ));
            }
        }
        Ok(())
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Runs ordinary prefill and publishes an additive capture in the same transaction.
    /// Partitioned callers supply an architecture-authored exact bundle; every rank
    /// collects source tensors or matching placeholders before any transfer.
    /// Input admission errors participate in readiness before bundle resolution
    /// or model execution; a failed caller may supply no bundle.
    pub fn prefill_input_with_capture<'a, O, C, F, Finish>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        bundle: Option<&ForwardCaptureBundle>,
        collect: F,
        assemble: Finish,
    ) -> Result<(B::Tensor, C), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        F: FnOnce(&A::ForwardContext) -> Result<Vec<B::Tensor>, A::Error>,
        Finish: FnOnce(&A::ForwardContext, Vec<B::Tensor>) -> Result<C, A::Error>,
    {
        self.capture_input(
            input,
            ExpertPass::Prefill,
            context,
            observer,
            bundle,
            collect,
            assemble,
        )
    }
    /// Runs cached decoding through the same capture and state-commit transaction.
    pub fn decode_input_with_capture<'a, O, C, F, Finish>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        bundle: Option<&ForwardCaptureBundle>,
        collect: F,
        assemble: Finish,
    ) -> Result<(B::Tensor, C), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        F: FnOnce(&A::ForwardContext) -> Result<Vec<B::Tensor>, A::Error>,
        Finish: FnOnce(&A::ForwardContext, Vec<B::Tensor>) -> Result<C, A::Error>,
    {
        self.capture_input(
            input,
            ExpertPass::Decode,
            context,
            observer,
            bundle,
            collect,
            assemble,
        )
    }
    fn capture_input<'a, O, C, F, Finish>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        bundle: Option<&ForwardCaptureBundle>,
        collect: F,
        assemble: Finish,
    ) -> Result<(B::Tensor, C), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        F: FnOnce(&A::ForwardContext) -> Result<Vec<B::Tensor>, A::Error>,
        Finish: FnOnce(&A::ForwardContext, Vec<B::Tensor>) -> Result<C, A::Error>,
    {
        self.ensure_commit_resolved()?;
        if D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT {
            return Err(ReplicatedTextSessionError::Contract(
                "partitioned capture requires bounded input-preparation agreement".into(),
            ));
        }
        self.with_observation_transaction(observer, |session, observer| {
            // Input admission can fail before invocation geometry exists. Agree
            // that result before comparing any geometry-dependent bundle digest.
            let prepared = input.map_err(ReplicatedTextSessionError::Architecture);
            let readiness = D::agree_distributed_phase(
                &mut session.execution,
                crate::DistributedExecutionPhase::InputPreparation,
                prepared.is_ok(),
                context,
            )
            .map_err(widen_infallible)
            .inspect_err(|_| {
                session.control_fence = Some(crate::DistributedExecutionPhase::InputPreparation)
            });
            let input = match (prepared, readiness) {
                (Ok(input), Ok(true)) => input,
                (Err(error), _) | (_, Err(error)) => return Err(error),
                (Ok(_), Ok(false)) => {
                    return Err(ReplicatedTextSessionError::Contract(
                        "another rank failed capture input preparation".into(),
                    ))
                }
            };
            let local = if D::PARTITIONED_SESSION
                && (bundle.is_none() || !D::DISTRIBUTED_DESCRIPTOR_AGREEMENT)
            {
                Err(ReplicatedTextSessionError::Contract(
                    "partitioned capture requires an exact bundle and descriptor agreement".into(),
                ))
            } else {
                Ok(())
            };
            let mut descriptor = bundle.map_or([0; 8], |plan| plan.descriptor);
            descriptor[6] = session.next_commit_epoch.value();
            descriptor[7] = if matches!(pass, ExpertPass::Prefill) {
                0
            } else {
                1
            };
            let phase = crate::DistributedExecutionPhase::PredictionTargetCapture;
            let agreed = D::agree_distributed_descriptor(
                &mut session.execution,
                phase,
                local.is_ok(),
                descriptor,
                context,
            )
            .map_err(widen_infallible)
            .inspect_err(|_| session.control_fence = Some(phase));
            local?;
            if !agreed? {
                return Err(ReplicatedTextSessionError::Contract(
                    "capture bundle identity or readiness differs across ranks".into(),
                ));
            }
            let (output, checkpoint, forward) =
                session.execute_input_before_publication(input, pass, context, observer)?;
            let captured = collect(&forward)
                .map_err(ReplicatedTextSessionError::Architecture)
                .and_then(|values| {
                    if let Some(plan) = bundle {
                        plan.validate(&values)
                            .map_err(ReplicatedTextSessionError::Contract)?;
                    }
                    Ok(values)
                });
            // Local collection/geometry failure must settle before any peer enters broadcast.
            let agreement = D::agree_distributed_phase(
                &mut session.execution,
                phase,
                captured.is_ok(),
                context,
            )
            .map_err(widen_infallible);
            let mut values = match (captured, agreement) {
                (Ok(values), Ok(true)) => values,
                (Err(error), _) | (_, Err(error)) => {
                    return session.rollback_failure(checkpoint, error, context)
                }
                (Ok(_), Ok(false)) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Contract(
                            "another rank failed capture collection".into(),
                        ),
                        context,
                    )
                }
            };
            // Complete the ordinary output publication on every rank before
            // substituting capture destinations. Otherwise a nonproducer's
            // placeholder can drop lazy TP predecessors still needed by its peer.
            let (output, checkpoint, forward) = session
                .publish_observed_output_transaction(output, checkpoint, forward, context)?;
            if let Some(plan) = bundle {
                for (slot, value) in plan.slots.iter().zip(&mut values) {
                    let published = D::publish_capture_tensor(
                        &mut session.execution,
                        value.clone(),
                        slot,
                        context,
                    )
                    .map_err(widen_infallible);
                    let agreement = D::agree_distributed_phase(
                        &mut session.execution,
                        crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
                        published.is_ok(),
                        context,
                    )
                    .map_err(widen_infallible);
                    *value = match (published, agreement) {
                        (Ok(value), Ok(true)) => value,
                        (Err(error), _) | (_, Err(error)) => {
                            return session.rollback_failure(checkpoint, error, context)
                        }
                        (Ok(_), Ok(false)) => {
                            return session.rollback_failure(
                                checkpoint,
                                ReplicatedTextSessionError::Contract(
                                    "another rank failed capture publication".into(),
                                ),
                                context,
                            )
                        }
                    };
                }
            }
            let captured = (|| {
                if let Some(plan) = bundle {
                    plan.validate(&values)
                        .map_err(ReplicatedTextSessionError::Contract)?;
                }
                assemble(&forward, values.clone()).map_err(ReplicatedTextSessionError::Architecture)
            })();
            let agreement = D::agree_distributed_phase(
                &mut session.execution,
                phase,
                captured.is_ok(),
                context,
            )
            .map_err(widen_infallible);
            let captured = match (captured, agreement) {
                (Ok(value), Ok(true)) => value,
                (Err(error), _) | (_, Err(error)) => {
                    return session.rollback_failure(checkpoint, error, context)
                }
                (Ok(_), Ok(false)) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Contract(
                            "another rank failed capture assembly".into(),
                        ),
                        context,
                    )
                }
            };
            let retained = values.iter().collect::<Vec<_>>();
            session
                .publish_with_context(
                    output, checkpoint, forward, context, observer, true, &retained,
                )
                .map(|(output, _)| (output, captured))
        })
    }
}
