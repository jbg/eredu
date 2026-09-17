//! One semantic prediction invocation, shared by execution and cold equations.

use super::{
    MaterializedPredictionExecutor, PredictionExtensionMaterializer, PredictionOperationInvoker,
};
use eredu_nn::{
    BlockwiseAttentionBackend, DistributedNeuralBackend, GroupedNeuralBackend, HyperNeuralBackend,
};

/// Actual borrowed or projected operands for one selected prediction invocation.
/// This describes equations only; it grants no occurrence, source or completion
/// authority. Target and prediction frontiers remain with the invocation owner.
#[derive(Debug, Clone, Copy)]
pub enum PredictionEquation<T> {
    /// Seeds prediction state from the actual completed target capture.
    Prefill {
        /// Full target capture consumed by architecture-specific seeding.
        target_capture: T,
        /// Exact shifted/prefixed hidden operand supplied by the shared driver.
        hidden: T,
        /// Exact shifted token sequence supplied by the shared driver.
        tokens: T,
    },
    /// Executes one architecture proposal depth and constructs its scalar token.
    Sequential {
        /// Actual capture or preceding prediction hidden output.
        hidden: T,
        /// Actual sampled token ID.
        token: u32,
        /// Architecture proposal depth, not physical module ordinal.
        depth: usize,
    },
    /// Executes the architecture's fused proposal from one scalar anchor.
    Fused {
        /// Actual committed anchor ID.
        anchor: u32,
        /// Selected number of proposal positions.
        capacity: usize,
    },
    /// Advances prediction state through the committed target replay.
    Replay {
        /// Actual committed target captures.
        captures: T,
        /// Actual committed replay token sequence.
        tokens: T,
    },
}

impl<T> PredictionEquation<T> {
    /// Borrows the same operands without cloning any tensor or container.
    pub fn as_ref(&self) -> PredictionEquation<&T> {
        match self {
            Self::Prefill {
                target_capture,
                hidden,
                tokens,
            } => PredictionEquation::Prefill {
                target_capture,
                hidden,
                tokens,
            },
            Self::Sequential {
                hidden,
                token,
                depth,
            } => PredictionEquation::Sequential {
                hidden,
                token: *token,
                depth: *depth,
            },
            Self::Fused { anchor, capacity } => PredictionEquation::Fused {
                anchor: *anchor,
                capacity: *capacity,
            },
            Self::Replay { captures, tokens } => PredictionEquation::Replay { captures, tokens },
        }
    }

    /// Projects actual operands in semantic order through one caller-owned
    /// alias collector. No destination or temporary collection is allocated.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> Result<U, E>,
    ) -> Result<PredictionEquation<U>, E> {
        Ok(match self {
            Self::Prefill {
                target_capture,
                hidden,
                tokens,
            } => PredictionEquation::Prefill {
                target_capture: map(target_capture)?,
                hidden: map(hidden)?,
                tokens: map(tokens)?,
            },
            Self::Sequential {
                hidden,
                token,
                depth,
            } => PredictionEquation::Sequential {
                hidden: map(hidden)?,
                token,
                depth,
            },
            Self::Fused { anchor, capacity } => PredictionEquation::Fused { anchor, capacity },
            Self::Replay { captures, tokens } => PredictionEquation::Replay {
                captures: map(captures)?,
                tokens: map(tokens)?,
            },
        })
    }
}

/// Actual semantic output roots, before the caller's existing logits-row tail.
pub enum PredictionEquationOutput<T> {
    /// Only the actual mutated prediction state remains.
    StateOnly,
    /// Both logits and hidden state remain live through completion.
    Sequential {
        /// Raw vocabulary scores before the caller's row tail.
        logits: T,
        /// Raw hidden output used by the next proposal.
        hidden: T,
    },
    /// The selected fused strategy may decline to produce a proposal.
    Fused(Option<T>),
}
impl<T> PredictionEquationOutput<T> {
    /// Visits every raw output alias without constructing a root collection.
    pub fn visit_roots(&self, mut visit: impl FnMut(&T)) {
        match self {
            Self::StateOnly | Self::Fused(None) => {}
            Self::Sequential { logits, hidden } => {
                visit(logits);
                visit(hidden);
            }
            Self::Fused(Some(logits)) => visit(logits),
        }
    }
}

/// Executes the same selected architecture methods used by ordinary prediction.
/// Scalar-token construction happens inside this invocation. The caller supplies
/// the ordinary token worker, observation transaction and completion protocol.
/// With no observer a fused equation still uses the immutable fused method;
/// observed fused execution receives the caller's temporary mutable lane.
#[allow(clippy::too_many_arguments)]
pub fn execute_prediction_equation<A, B, S, N, P, I>(
    equation: PredictionEquation<&B::Tensor>,
    extension: &mut P,
    invoker: &mut I,
    lane: &mut P::LaneState,
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<B::Tensor, eredu_nn::Error>>,
    make_token: impl FnOnce(u32) -> Result<B::Tensor, I::Error>,
) -> Result<PredictionEquationOutput<B::Tensor>, I::Error>
where
    B: BlockwiseAttentionBackend
        + DistributedNeuralBackend
        + GroupedNeuralBackend
        + HyperNeuralBackend,
    S: eredu_runtime::RuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = eredu_nn::Error>,
    N: PredictionExtensionMaterializer<B>,
    P: MaterializedPredictionExecutor<A, B, N>,
    I: PredictionOperationInvoker<A, B, S>,
{
    Ok(match equation {
        PredictionEquation::Prefill {
            target_capture,
            hidden,
            tokens,
        } => {
            extension.prefill_observed::<S, I>(
                invoker,
                target_capture,
                hidden,
                tokens,
                lane,
                observer,
            )?;
            PredictionEquationOutput::StateOnly
        }
        PredictionEquation::Sequential {
            hidden,
            token,
            depth,
        } => {
            let token = make_token(token)?;
            let (logits, hidden) = extension
                .logits_observed::<S, I>(invoker, hidden, &token, depth, lane, observer)?;
            PredictionEquationOutput::Sequential { logits, hidden }
        }
        PredictionEquation::Fused { anchor, capacity } => {
            let anchor = make_token(anchor)?;
            let output = match observer {
                Some(observer) => extension.fused_logits_observed::<S, I>(
                    invoker,
                    &anchor,
                    capacity,
                    lane,
                    Some(observer),
                )?,
                None => extension.fused_logits::<S, I>(invoker, &anchor, capacity, lane)?,
            };
            PredictionEquationOutput::Fused(output)
        }
        PredictionEquation::Replay { captures, tokens } => {
            extension.advance_observed::<S, I>(invoker, captures, tokens, lane, observer)?;
            PredictionEquationOutput::StateOnly
        }
    })
}

/// Fixed source-geometry refusal. Reading a frontier never allocates or grants
/// source/occurrence authority; native and workspace callers retain their own
/// typed error transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PredictionFrontierError {
    /// Selected depth, member population or group range differs from the lane.
    #[error("prediction equation state membership differs from its source")]
    Membership,
    /// Stateful members consumed by the same equation have different positions.
    #[error("prediction equation state frontiers differ")]
    Different,
    /// The selected operation has no stateful source coordinate.
    #[error("prediction equation has no stateful source frontier")]
    Empty,
    /// A cache reports an invalid negative coordinate.
    #[error("prediction equation has a negative source frontier")]
    Negative,
}

pub(super) fn selected_members<T>(
    equation: &PredictionEquation<T>,
    count: usize,
) -> Result<std::ops::Range<usize>, PredictionFrontierError> {
    match equation {
        PredictionEquation::Sequential { depth, .. } => {
            if *depth >= count {
                return Err(PredictionFrontierError::Membership);
            }
            Ok(*depth
                ..depth
                    .checked_add(1)
                    .ok_or(PredictionFrontierError::Membership)?)
        }
        _ => Ok(0..count),
    }
}

pub(super) fn source_frontier(
    positions: impl Iterator<Item = Option<i32>>,
    selected: std::ops::Range<usize>,
    expected: usize,
) -> Result<u64, PredictionFrontierError> {
    if selected.start >= selected.end || selected.end > expected {
        return Err(PredictionFrontierError::Membership);
    }
    let mut count = 0usize;
    let mut frontier = None;
    for (index, position) in positions.enumerate() {
        count = index
            .checked_add(1)
            .ok_or(PredictionFrontierError::Membership)?;
        if selected.contains(&index) {
            if let Some(position) = position {
                if frontier.is_some_and(|prior| prior != position) {
                    return Err(PredictionFrontierError::Different);
                }
                frontier = Some(position);
            }
        }
    }
    if count != expected {
        return Err(PredictionFrontierError::Membership);
    }
    u64::try_from(frontier.ok_or(PredictionFrontierError::Empty)?)
        .map_err(|_| PredictionFrontierError::Negative)
}

#[cfg(test)]
mod frontier_tests {
    use super::*;
    #[test]
    fn proposal_frontier_uses_selected_depth_and_complete_group_membership() {
        let proposal = PredictionEquation::Sequential {
            hidden: (),
            token: 7,
            depth: 1,
        };
        let selected = selected_members(&proposal, 2).unwrap();
        assert_eq!(
            source_frontier([Some(8), Some(7)].into_iter(), selected, 2),
            Ok(7)
        );
        assert_eq!(
            source_frontier([Some(8), Some(7)].into_iter(), 0..2, 2),
            Err(PredictionFrontierError::Different)
        );
        assert_eq!(
            source_frontier([Some(8), None, Some(7), Some(7)].into_iter(), 2..4, 4),
            Ok(7)
        );
        assert_eq!(
            source_frontier([Some(8), None, Some(7)].into_iter(), 2..4, 4),
            Err(PredictionFrontierError::Membership)
        );
        assert_eq!(
            source_frontier([None, None].into_iter(), 0..2, 2),
            Err(PredictionFrontierError::Empty)
        );
        assert_eq!(
            source_frontier([Some(-1)].into_iter(), 0..1, 1),
            Err(PredictionFrontierError::Negative)
        );
    }
}
