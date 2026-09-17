//! Source-aware immutable fused context using the existing host value owner.
use super::*;
use crate::external_assistant::ExternalOperationResult;
use crate::speculative_execution::{EmbeddedPredictionTensor, PreparedEmbeddedEvidence};
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, SpeculativeValues};
use std::mem::{size_of, size_of_val};

/// Immutable committed context, preserving ordinary clone semantics or sharing
/// the exact prepared vector and its independently retained completion source.
#[derive(Clone)]
pub struct SharedDFlashContext<T: Clone> {
    values: ContextValues<T>,
    evidence: Option<PreparedEmbeddedEvidence>,
}
#[derive(Clone)]
enum ContextValues<T: Clone> {
    Ordinary(DFlashContext<T>),
    Prepared(SpeculativeValues<DFlashContext<T>>),
}
impl<T: Clone> std::ops::Deref for SharedDFlashContext<T> {
    type Target = DFlashContext<T>;
    fn deref(&self) -> &Self::Target {
        match &self.values {
            ContextValues::Ordinary(value) => value,
            ContextValues::Prepared(value) => &value[0],
        }
    }
}
impl<T: Clone> From<DFlashContext<T>> for SharedDFlashContext<T> {
    fn from(value: DFlashContext<T>) -> Self {
        Self {
            values: ContextValues::Ordinary(value),
            evidence: None,
        }
    }
}
impl<T: Clone> SharedDFlashContext<T> {
    pub(super) fn evidence(&self) -> Option<&PreparedEmbeddedEvidence> {
        self.evidence.as_ref()
    }
}
type Architecture = MuseGlimmerAssistantArchitecture;
fn controls<M: ExternalAssistantExecutionMechanisms<Architecture>, const N: usize>(
    parts: [usize; N],
    context: M::Context<'_>,
) -> Result<HostPreparationAuthority, M::Error> {
    M::state_host_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add),
        context,
    )
}

pub(super) fn freeze<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    value: ExternalOperationResult<DFlashContext<M::Tensor>>,
    context: M::Context<'_>,
) -> Result<SharedDFlashContext<M::Tensor>, M::Error> {
    let _host = M::state_host_metadata(
        Some(std::mem::size_of::<(
            SharedDFlashContext<M::Tensor>,
            Result<SharedDFlashContext<M::Tensor>, M::Error>,
        )>()),
        context,
    )?;
    let mut values = M::state_buffer(1, context)?;
    values
        .try_push(value.output)
        .map_err(|_| M::state_refusal(context))?;
    Ok(SharedDFlashContext {
        values: ContextValues::Prepared(M::freeze_state_values(values, context)?),
        evidence: value.evidence,
    })
}
pub(super) fn assemble<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    assistant: &mut M::Assistant,
    states: &[M::Tensor],
    evidence: Option<&PreparedEmbeddedEvidence>,
    context: M::Context<'_>,
) -> Result<ExternalOperationResult<M::Tensor>, M::Error> {
    let _host = controls::<M, 4>(
        [
            size_of::<ExternalOperationResult<M::Tensor>>(),
            size_of::<Result<ExternalOperationResult<M::Tensor>, M::Error>>(),
            size_of::<super::super::assistant::invocation::RawContextArguments<'_, M::Tensor>>(),
            size_of::<(M::Context<'_>, &[&PreparedEmbeddedEvidence])>(),
        ],
        context,
    )?;
    let proofs = evidence.as_slice();
    let source = M::source_context(context, proofs)?;
    M::assistant_operation_with_evidence::<super::super::assistant::invocation::RawContext>(
        assistant,
        super::super::assistant::invocation::RawContextArguments {
            previous: None,
            states,
        },
        source,
    )
}
pub(super) fn update<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    assistant: &mut M::Assistant,
    previous: Option<&SharedDFlashContext<M::Tensor>>,
    pending: &M::Tensor,
    evidence: Option<&PreparedEmbeddedEvidence>,
    absolute_end: i32,
    context: M::Context<'_>,
) -> Result<SharedDFlashContext<M::Tensor>, M::Error> {
    let _host = controls::<M, 5>(
        [
            size_of::<SharedDFlashContext<M::Tensor>>(),
            size_of::<ExternalOperationResult<DFlashContext<M::Tensor>>>(),
            size_of::<Result<SharedDFlashContext<M::Tensor>, M::Error>>(),
            size_of::<super::super::assistant::invocation::UpdateContextArguments<'_, M::Tensor>>(),
            size_of::<(M::Context<'_>, SpeculativeBuffer<&PreparedEmbeddedEvidence>)>(),
        ],
        context,
    )?;
    let mut proofs = M::state_buffer(2, context)?;
    if let Some(proof) = evidence {
        proofs
            .try_push(proof)
            .map_err(|_| M::state_refusal(context))?;
    }
    if let Some(proof) = previous.and_then(|value| value.evidence()) {
        proofs
            .try_push(proof)
            .map_err(|_| M::state_refusal(context))?;
    }
    let source = M::source_context(context, &proofs)?;
    let output =
        M::assistant_operation_with_evidence::<super::super::assistant::invocation::UpdateContext>(
            assistant,
            super::super::assistant::invocation::UpdateContextArguments {
                previous: previous.map(|value| &**value),
                pending,
                absolute_end,
            },
            source,
        )?;
    drop(proofs);
    freeze::<M>(output, context)
}
pub(super) fn proposal<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    assistant: &mut M::Assistant,
    embeddings: &M::Tensor,
    evidence: Option<&PreparedEmbeddedEvidence>,
    committed: &SharedDFlashContext<M::Tensor>,
    absolute_end: i32,
    context: M::Context<'_>,
) -> Result<ExternalOperationResult<M::Tensor>, M::Error> {
    let _host = controls::<M, 4>(
        [
            size_of::<ExternalOperationResult<M::Tensor>>(),
            size_of::<Result<ExternalOperationResult<M::Tensor>, M::Error>>(),
            size_of::<super::super::assistant::invocation::FusedProposalArguments<'_, M::Tensor>>(),
            size_of::<(M::Context<'_>, SpeculativeBuffer<&PreparedEmbeddedEvidence>)>(),
        ],
        context,
    )?;
    let mut proofs = M::state_buffer(2, context)?;
    if let Some(proof) = evidence {
        proofs
            .try_push(proof)
            .map_err(|_| M::state_refusal(context))?;
    }
    if let Some(proof) = committed.evidence() {
        proofs
            .try_push(proof)
            .map_err(|_| M::state_refusal(context))?;
    }
    let source = M::source_context(context, &proofs)?;
    M::assistant_operation_with_evidence::<super::super::assistant::invocation::FusedProposal>(
        assistant,
        super::super::assistant::invocation::FusedProposalArguments {
            embeddings,
            committed,
            absolute_end,
        },
        source,
    )
}
pub(super) fn embeddings<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    target: &mut M::Target,
    ids: &[u32],
    context: M::Context<'_>,
) -> Result<EmbeddedPredictionTensor<M::Tensor>, M::Error> {
    let _host = controls::<M, 4>(
        [
            size_of::<EmbeddedPredictionTensor<M::Tensor>>(),
            size_of::<Result<EmbeddedPredictionTensor<M::Tensor>, M::Error>>(),
            size_of::<crate::composite_execution::ExternalPredictionTargetOperation<'_, M::Tensor>>(
            ),
            size_of::<(M::Context<'_>, Option<&PreparedEmbeddedEvidence>)>(),
        ],
        context,
    )?;
    let tokens = M::target_tokens_with_source(ids, context)?;
    let evidence = tokens.evidence();
    let source = M::source_context(context, evidence.as_slice())?;
    M::target_operation_with_source(
        target,
        crate::composite_execution::ExternalPredictionTargetOperation::TokenEmbeddings(&tokens),
        source,
    )
}
pub(super) fn logits<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    target: &mut M::Target,
    states: &M::Tensor,
    evidence: Option<&PreparedEmbeddedEvidence>,
    context: M::Context<'_>,
) -> Result<EmbeddedPredictionTensor<M::Tensor>, M::Error> {
    let _host = controls::<M, 3>(
        [
            size_of::<Result<EmbeddedPredictionTensor<M::Tensor>, M::Error>>(),
            size_of::<crate::composite_execution::ExternalPredictionTargetOperation<'_, M::Tensor>>(
            ),
            size_of::<(M::Context<'_>, Option<&PreparedEmbeddedEvidence>)>(),
        ],
        context,
    )?;
    let source = M::source_context(context, evidence.as_slice())?;
    M::target_operation_with_source(
        target,
        crate::composite_execution::ExternalPredictionTargetOperation::ProjectLogits(states),
        source,
    )
}

pub(super) fn control_copy<M: ExternalAssistantExecutionMechanisms<Architecture>>(
    state: &ExternalTargetState<M::Tensor>,
    context: M::Context<'_>,
) -> Result<ExternalTargetState<M::Tensor>, M::Error> {
    let _host = controls::<M, 5>(
        [
            size_of::<(
                ExternalTargetState<M::Tensor>,
                DFlashContext<M::Tensor>,
                ExternalOperationResult<M::Tensor>,
                Result<ExternalTargetState<M::Tensor>, M::Error>,
            )>(),
            size_of::<(
                ExternalOperationResult<M::Tensor>,
                ExternalOperationResult<M::Tensor>,
            )>(),
            size_of::<(
                SpeculativeBuffer<PreparedEmbeddedEvidence>,
                SpeculativeBuffer<&PreparedEmbeddedEvidence>,
                Vec<super::super::DFlashLayerContext<M::Tensor>>,
            )>(),
            size_of::<std::slice::Iter<'_, super::super::DFlashLayerContext<M::Tensor>>>(),
            size_of::<(
                M::Context<'_>,
                &SharedDFlashContext<M::Tensor>,
                &ExternalTargetState<M::Tensor>,
            )>(),
        ],
        context,
    )?;
    let pending = state
        .pending_context
        .as_ref()
        .map(|value| {
            M::control_copy_tensor_with_source(
                value,
                state.evidence.as_ref(),
                ExternalAssistantTensorPlacement::Target,
                context,
            )
        })
        .transpose()?;
    let draft_context = state
        .draft_context
        .as_ref()
        .map(|previous| {
            let count = previous.layers.len();
            let roots = count
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| M::state_refusal(context))?;
            let mut proofs = M::state_buffer(roots, context)?;
            let mut layers = M::state_vector(count, context)?;
            let encoded = M::control_copy_tensor_with_source(
                &previous.encoded,
                previous.evidence(),
                ExternalAssistantTensorPlacement::Draft,
                context,
            )?;
            if let Some(proof) = encoded.evidence {
                proofs
                    .try_push(proof)
                    .map_err(|_| M::state_refusal(context))?;
            }
            for layer in &previous.layers {
                let keys = M::control_copy_tensor_with_source(
                    &layer.keys,
                    previous.evidence(),
                    ExternalAssistantTensorPlacement::Draft,
                    context,
                )?;
                let values = M::control_copy_tensor_with_source(
                    &layer.values,
                    previous.evidence(),
                    ExternalAssistantTensorPlacement::Draft,
                    context,
                )?;
                if let Some(proof) = keys.evidence {
                    proofs
                        .try_push(proof)
                        .map_err(|_| M::state_refusal(context))?;
                }
                if let Some(proof) = values.evidence {
                    proofs
                        .try_push(proof)
                        .map_err(|_| M::state_refusal(context))?;
                }
                layers.push(super::super::DFlashLayerContext {
                    keys: keys.output,
                    values: values.output,
                });
            }
            let value = DFlashContext {
                encoded: encoded.output,
                layers,
                start: previous.start,
                end: previous.end,
            };
            let mut refs = M::state_buffer(proofs.len(), context)?;
            for proof in &proofs {
                refs.try_push(proof)
                    .map_err(|_| M::state_refusal(context))?;
            }
            let evidence = M::join_tensor_sources_at(
                |visit| {
                    visit(&value.encoded);
                    for layer in &value.layers {
                        visit(&layer.keys);
                        visit(&layer.values);
                    }
                },
                &refs,
                ExternalAssistantTensorPlacement::Draft,
                context,
            )?;
            freeze::<M>(
                ExternalOperationResult {
                    output: value,
                    evidence,
                },
                context,
            )
        })
        .transpose()?;
    let (pending_context, evidence) = match pending {
        Some(value) => (Some(value.output), value.evidence),
        None => (None, None),
    };
    Ok(ExternalTargetState {
        pending_context,
        draft_context,
        cache_len: state.cache_len,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Debug)]
    struct Value {
        number: u32,
        dropped: Arc<AtomicUsize>,
        cloned: Arc<AtomicUsize>,
    }
    impl Clone for Value {
        fn clone(&self) -> Self {
            self.cloned.fetch_add(1, Ordering::SeqCst);
            Self {
                number: self.number,
                dropped: self.dropped.clone(),
                cloned: self.cloned.clone(),
            }
        }
    }
    impl Drop for Value {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Host {
        dropped: Arc<AtomicUsize>,
        retired: Arc<AtomicUsize>,
    }
    impl Drop for Host {
        fn drop(&mut self) {
            assert_eq!(self.dropped.load(Ordering::SeqCst), 3);
            self.retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn prepared_fused_context_shares_all_values_and_retires_them_before_host() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let cloned = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicUsize::new(0));
        let value = |number| Value {
            number,
            dropped: dropped.clone(),
            cloned: cloned.clone(),
        };
        let host = HostPreparationAuthority::retain(Host {
            dropped: dropped.clone(),
            retired: retired.clone(),
        });
        let mut rows = SpeculativeBuffer::try_new_retained(1, host.clone()).unwrap();
        rows.try_push(DFlashContext {
            encoded: value(13),
            layers: vec![super::super::super::DFlashLayerContext {
                keys: value(29),
                values: value(43),
            }],
            start: 7,
            end: 10,
        })
        .unwrap();
        let source = SharedDFlashContext {
            values: ContextValues::Prepared(SpeculativeValues::from_prepared_buffer(rows, host)),
            evidence: None,
        };
        let saved = source.clone();
        assert_eq!(cloned.load(Ordering::SeqCst), 0);
        assert!(std::ptr::eq(&*source, &*saved));
        assert_eq!(
            (
                saved.encoded.number,
                saved.layers[0].keys.number,
                saved.layers[0].values.number
            ),
            (13, 29, 43)
        );
        assert_eq!((saved.start, saved.end), (7, 10));
        drop(source);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(saved);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
        let ordinary = SharedDFlashContext::from(DFlashContext {
            encoded: 1,
            layers: vec![super::super::super::DFlashLayerContext { keys: 2, values: 3 }],
            start: 0,
            end: 1,
        });
        let copy = ordinary.clone();
        assert!(!std::ptr::eq(&*ordinary, &*copy));
        assert_eq!(copy.layers[0].values, 3);
    }
}
