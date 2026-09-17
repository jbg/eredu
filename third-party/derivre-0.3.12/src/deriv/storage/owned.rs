//! One owning expression/memo boundary, reusing the existing borrowed workers.
mod copy;
pub use copy::PreparedExpressionCopyFailure;
mod relevance;
mod symbolic;
use super::{workspace, OperationError, Parts, Storage};
use crate::ast::{
    weights::storage as weights, ExprRef, ExprSet, PreparedExprError, PreparedExprSet,
};
pub use relevance::PreparedRelevanceError;
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
pub use symbolic::PreparedSymbolicError;
pub use weights::PreparedWeightError;

/// Complete local destination requirements; the prepared expression source is
/// already independently owned and is not counted again as a fresh allocation.
#[derive(Clone, Copy, Debug)]
pub struct PreparedExpressionRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
}
impl PreparedExpressionRequirements {
    /// Newly constructed memo, traversal and simplifier buffer bytes.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Local preparation and owning-handoff frames.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete local constructor requirement, excluding existing source storage.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
enum Cause {
    Relevance {
        cause: relevance::InitFailure,
        prepared: Option<super::Parts>,
        symbolic: Option<symbolic::Parts>,
        weights: Option<weights::Parts>,
    },
    Weights {
        cause: weights::InitFailure,
        prepared: Option<super::Parts>,
        symbolic: Option<symbolic::Parts>,
    },
    Overflow,
    Construction(super::Failure),
    Symbolic {
        cause: symbolic::InitFailure,
        prepared: Option<super::Parts>,
    },
}
impl fmt::Debug for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weights {
                cause,
                prepared,
                symbolic,
            } => f
                .debug_struct("Weights")
                .field("cause", cause)
                .field("retains_derivative", &prepared.is_some())
                .field("retains_symbolic", &symbolic.is_some())
                .finish(),
            Self::Relevance {
                cause,
                prepared,
                symbolic,
                weights,
            } => f
                .debug_struct("Relevance")
                .field("cause", cause)
                .field("retains_derivative", &prepared.is_some())
                .field("retains_symbolic", &symbolic.is_some())
                .field("retains_weights", &weights.is_some())
                .finish(),
            Self::Overflow => f.write_str("Overflow"),
            Self::Construction(cause) => f.debug_tuple("Construction").field(cause).finish(),
            Self::Symbolic { cause, prepared } => f
                .debug_struct("Symbolic")
                .field("cause", cause)
                .field("retains_derivative", &prepared.is_some())
                .finish(),
        }
    }
}
/// Failed constructor prefix followed by the same complete expression owner.
/// No ordinary expression alias or failed destination is silently discarded.
pub struct PreparedExpressionFailure {
    cause: Cause,
    source: PreparedExprSet,
}
impl fmt::Debug for PreparedExpressionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedExpressionFailure")
            .field("cause", &self.cause)
            .field("source", &self.source)
            .finish()
    }
}
impl fmt::Display for PreparedExpressionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("owning expression construction geometry overflow"),
            Cause::Construction(e) => fmt::Display::fmt(e, f),
            Cause::Symbolic { cause, .. } => fmt::Display::fmt(cause, f),
            Cause::Weights { cause, .. } => fmt::Display::fmt(cause, f),
            Cause::Relevance { cause, .. } => fmt::Display::fmt(cause, f),
        }
    }
}
impl std::error::Error for PreparedExpressionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Construction(e) => Some(e),
            Cause::Symbolic { cause, .. } => Some(cause),
            Cause::Weights { cause, .. } => Some(cause),
            Cause::Relevance { cause, .. } => Some(cause),
            _ => None,
        }
    }
}
impl PreparedExpressionFailure {
    /// Borrows the exact retained source, including completed mutations.
    pub fn expression_source(&self) -> &ExprSet {
        self.source.source()
    }
}
/// Move-only source plan. No caller can mutate its expression owner between
/// inspection, destination construction and the final owning handoff.
pub struct PreparedExpressionPlan {
    source: PreparedExprSet,
    requirements: PreparedExpressionRequirements,
}
impl fmt::Debug for PreparedExpressionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedExpressionPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
/// A complete finite expression source and its persistent derivative memo.
/// Internal parts are only storage; all calls borrow this same retained source.
/// This owner does not grant lexer, parser or controller execution admission.
pub struct PreparedExpressionMachine {
    parts: Option<Parts>,
    symbolic: Option<symbolic::Parts>,
    weights: Option<weights::Parts>,
    relevance: Option<relevance::Parts>,
    source: PreparedExprSet,
    failed: bool,
}
impl fmt::Debug for PreparedExpressionMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedExpressionMachine")
            .field("source", &self.source)
            .field("retains_parts", &self.parts.is_some())
            .finish()
    }
}
/// A checked derivative stopped while the machine retained its source and
/// completed memo/expression prefix. The failed scope cannot be reused.
#[derive(Debug)]
pub struct PreparedExpressionOperationError {
    cause: workspace::OperationError<OperationError<PreparedExprError>>,
}
impl fmt::Display for PreparedExpressionOperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PreparedExpressionOperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl PreparedExpressionPlan {
    /// Fixed source inspection/owning handoff frames, before walking the source.
    pub fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<PreparedExpressionMachine>(),
            size_of::<PreparedExpressionFailure>(),
            size_of::<PreparedExpressionRequirements>(),
            size_of::<PreparedExprSet>(),
            size_of::<Parts>(),
            size_of::<Option<Parts>>(),
            size_of::<Cause>(),
            weights::Plan::inspection_control_bytes()?,
            relevance::Plan::inspection_control_bytes()?,
            size_of::<weights::Parts>(),
            size_of::<Result<weights::Parts, weights::InitFailure>>(),
            size_of::<Result<Self, PreparedExpressionFailure>>(),
            size_of::<Result<PreparedExpressionMachine, PreparedExpressionFailure>>(),
            size_of::<Result<Parts, super::Failure>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Inspects the actual finite owner without constructing any destination.
    pub fn prepare(mut source: PreparedExprSet) -> Result<Self, PreparedExpressionFailure> {
        let result = (|| {
            let plan = source
                .derivative_scope_plan()
                .map_err(Cause::Construction)?;
            let base = plan.requirements();
            drop(plan);
            let symbolic = symbolic::Plan::prepare(source.source_mut())
                .map_err(|cause| Cause::Symbolic {
                    cause,
                    prepared: None,
                })?
                .requirements();
            let weights = weights::Plan::prepare(source.source())
                .map_err(|cause| Cause::Weights {
                    cause,
                    prepared: None,
                    symbolic: None,
                })?
                .requirements();
            let relevance = relevance::Plan::prepare(source.source())
                .map_err(|cause| Cause::Relevance {
                    cause,
                    prepared: None,
                    symbolic: None,
                    weights: None,
                })?
                .requirements();
            let buffers = base
                .buffers
                .checked_add(symbolic.buffers)
                .and_then(|n| n.checked_add(weights.buffers))
                .and_then(|n| n.checked_add(relevance.buffers))
                .ok_or(Cause::Overflow)?;
            let controls = base
                .controls
                .checked_add(symbolic.controls)
                .and_then(|n| n.checked_add(weights.controls))
                .and_then(|n| n.checked_add(relevance.controls))
                .and_then(|n| n.checked_add(Self::inspection_control_bytes()?))
                .ok_or(Cause::Overflow)?;
            let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
            Ok(PreparedExpressionRequirements {
                buffers,
                controls,
                total,
            })
        })();
        match result {
            Ok(requirements) => Ok(Self {
                source,
                requirements,
            }),
            Err(cause) => Err(PreparedExpressionFailure { cause, source }),
        }
    }
    /// Exact new buffer and local constructor requirements for this source.
    pub fn requirements(&self) -> PreparedExpressionRequirements {
        self.requirements
    }
    /// Constructs the same scoped destinations and moves their parts beside the
    /// original source; callers reserve `requirements()` before this operation.
    pub fn compile(mut self) -> Result<PreparedExpressionMachine, PreparedExpressionFailure> {
        let result = (|| {
            let plan = self.source.derivative_scope_plan()?;
            let storage = plan.compile()?;
            Ok::<_, super::Failure>(storage.into_parts())
        })();
        let parts = match result {
            Ok(parts) => parts,
            Err(cause) => {
                return Err(PreparedExpressionFailure {
                    cause: Cause::Construction(cause),
                    source: self.source,
                });
            }
        };
        let symbolic =
            symbolic::Plan::prepare(self.source.source_mut()).and_then(|plan| plan.compile());
        let symbolic = match symbolic {
            Ok(symbolic) => symbolic,
            Err(cause) => {
                return Err(PreparedExpressionFailure {
                    cause: Cause::Symbolic {
                        cause,
                        prepared: Some(parts),
                    },
                    source: self.source,
                })
            }
        };
        let weights = weights::Plan::prepare(self.source.source()).and_then(|plan| plan.compile());
        let weights = match weights {
            Ok(weights) => weights,
            Err(cause) => {
                return Err(PreparedExpressionFailure {
                    cause: Cause::Weights {
                        cause,
                        prepared: Some(parts),
                        symbolic: Some(symbolic),
                    },
                    source: self.source,
                })
            }
        };
        let relevance = relevance::Plan::prepare(self.source.source()).and_then(|p| p.compile());
        match relevance {
            Ok(relevance) => Ok(PreparedExpressionMachine {
                parts: Some(parts),
                symbolic: Some(symbolic),
                weights: Some(weights),
                relevance: Some(relevance),
                source: self.source,
                failed: false,
            }),
            Err(cause) => Err(PreparedExpressionFailure {
                cause: Cause::Relevance {
                    cause,
                    prepared: Some(parts),
                    symbolic: Some(symbolic),
                    weights: Some(weights),
                },
                source: self.source,
            }),
        }
    }
}
impl PreparedExpressionMachine {
    /// Same conservative ordinary prefix-containment policy and pair cache.
    /// Fuel refusal is recoverable exactly as required by subsumption; actual
    /// funding/source/equation failures remain terminal and preserve custody.
    pub fn is_contained_in_prefixes<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        small: ExprRef,
        big: ExprRef,
        max_fuel: u64,
        cache_failures: bool,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        if self.failed {
            return Err(PreparedRelevanceError::Failed);
        }
        self.failed = true;
        reserve(
            relevance::containment_operation_control_bytes::<F, E>()
                .ok_or(PreparedRelevanceError::Overflow)?,
        )
        .map_err(PreparedRelevanceError::Funding)?;
        let weights = self
            .weights
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let relevance = self
            .relevance
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let result = relevance.contained(
            self.source.source_mut(),
            &mut self.parts,
            &mut self.symbolic,
            weights,
            small,
            big,
            max_fuel,
            cache_failures,
            reserve,
        );
        if result.is_ok() || matches!(&result, Err(PreparedRelevanceError::Fuel { .. })) {
            self.failed = false;
        }
        result
    }
    /// Shared complete non-emptiness entry, including concatenation filtering,
    /// repeat probes and the existing DFS, under one ordinary expression-cost
    /// ceiling. Source/destination errors preserve their prefix and fence reuse.
    pub fn is_non_empty<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        root: ExprRef,
        max_fuel: u64,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        if self.failed {
            return Err(PreparedRelevanceError::Failed);
        }
        self.failed = true;
        reserve(
            relevance::entry_operation_control_bytes::<F, E>()
                .ok_or(PreparedRelevanceError::Overflow)?,
        )
        .map_err(PreparedRelevanceError::Funding)?;
        let derivative = self.parts.as_mut().ok_or(PreparedRelevanceError::Failed)?;
        let weights = self
            .weights
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let relevance = self
            .relevance
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let result = relevance.non_empty(
            self.source.source_mut(),
            derivative,
            &mut self.symbolic,
            weights,
            root,
            max_fuel,
            reserve,
        );
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
    /// Executes the shared relevance DFS stage over the same retained source.
    /// The ordinary entry's concat/repeat optimizations and containment wrapper
    /// are separate producers; this raw stage grants no controller admission.
    /// Each reached variable destination is reserved before construction.
    pub fn relevance_walk<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        root: ExprRef,
        max_fuel: u64,
        reserve: &F,
    ) -> Result<bool, PreparedRelevanceError<E>> {
        if self.failed {
            return Err(PreparedRelevanceError::Failed);
        }
        self.failed = true;
        reserve(
            relevance::operation_control_bytes::<F, E>().ok_or(PreparedRelevanceError::Overflow)?,
        )
        .map_err(PreparedRelevanceError::Funding)?;
        let derivative = self.parts.as_mut().ok_or(PreparedRelevanceError::Failed)?;
        let weights = self
            .weights
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let relevance = self
            .relevance
            .as_mut()
            .ok_or(PreparedRelevanceError::Failed)?;
        let result = relevance.execute(
            self.source.source_mut(),
            derivative,
            &mut self.symbolic,
            weights,
            root,
            max_fuel,
            reserve,
        );
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
    /// Local frames for either weight query. Reached destination growth consumes
    /// the same separately bound expression-source funding owner.
    pub fn weight_operation_control_bytes() -> Option<usize> {
        let frames = [
            weights::operation_control_bytes()?,
            size_of::<&mut Self>(),
            size_of::<Result<u32, PreparedWeightError>>(),
            size_of::<Result<bool, PreparedWeightError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Shared ordinary weight equations over the same retained finite arena.
    /// Failure preserves completed cache entries and fences the whole machine.
    pub fn weight(&mut self, root: ExprRef) -> Result<u32, PreparedWeightError> {
        if self.failed {
            return Err(PreparedWeightError::Failed);
        }
        self.failed = true;
        let result = self
            .weights
            .as_mut()
            .ok_or(PreparedWeightError::Failed)?
            .attributes(self.source.source(), root)
            .map(|v| v.0);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
    /// Inherited repeat attribute from the same shared weight traversal/cache.
    pub fn has_repeat(&mut self, root: ExprRef) -> Result<bool, PreparedWeightError> {
        if self.failed {
            return Err(PreparedWeightError::Failed);
        }
        self.failed = true;
        let result = self
            .weights
            .as_mut()
            .ok_or(PreparedWeightError::Failed)?
            .has_repeat(self.source.source(), root);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
    /// Read-only access to the same IDs, metadata and complete finite source.
    pub fn source(&self) -> &ExprSet {
        self.source.source()
    }
    /// Completed or attempted shared derivative-node evaluations, including the
    /// retained failure prefix; memo hits do not add evaluations. `None` means an
    /// interrupted invocation lost its local parts, never a refunded zero count.
    pub fn num_derivatives(&self) -> Option<usize> {
        self.parts.as_ref().map(|parts| parts.num_deriv)
    }
    /// Local typed operation frames; the enclosing grammar operation still prices
    /// its transport and recursive constructor frames before invoking this worker.
    pub fn operation_control_bytes() -> Option<usize> {
        let frames = [
            Storage::prepared_operation_control_bytes()?,
            size_of::<(&ExprSet, &mut Parts)>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<Result<(usize, usize, usize), PreparedExprError>>(),
            size_of::<Result<(), PreparedExprError>>(),
            size_of::<&mut Self>(),
            size_of::<Option<Parts>>(),
            size_of::<Parts>(),
            size_of::<Storage<'_>>(),
            size_of::<PreparedExpressionOperationError>(),
            size_of::<Result<ExprRef, PreparedExpressionOperationError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Uses the shared finite derivative worker and preserves its memo across calls.
    /// No other expression source can be supplied to this owning boundary.
    pub fn derivative(
        &mut self,
        root: ExprRef,
        byte: u8,
    ) -> Result<ExprRef, PreparedExpressionOperationError> {
        if self.failed {
            return Err(PreparedExpressionOperationError {
                cause: workspace::OperationError::Process(OperationError::Failed),
            });
        }
        self.failed = true;
        let mut parts = self
            .parts
            .take()
            .ok_or_else(|| PreparedExpressionOperationError {
                cause: workspace::OperationError::Process(OperationError::Failed),
            })?;
        if let Err(cause) = parts.prepare_reached(self.source.source()) {
            self.parts = Some(parts);
            return Err(PreparedExpressionOperationError {
                cause: workspace::OperationError::Process(OperationError::Arena(cause)),
            });
        }
        let mut scope = parts.bind(self.source.source_mut());
        let result = scope.derivative_prepared(root, byte);
        self.parts = Some(scope.into_parts());
        if result.is_ok() {
            self.failed = false;
        }
        result.map_err(|cause| PreparedExpressionOperationError { cause })
    }
}

impl PreparedExpressionMachine {
    /// Actual shared symbolic node evaluations; unavailable only if an unwind
    /// lost local parts. Memo hits still reserve their actual output copy.
    pub fn num_symbolic_nodes(&self) -> Option<usize> {
        self.symbolic.as_ref().map(|parts| parts.nodes)
    }
    /// Uses the same symbolic equations, reserving every reached constructor and
    /// memo/value copy before allocation through the caller's exact host funding.
    /// The callback supplies no expression source or alternative evaluator.
    pub fn symbolic_derivative<F: Fn(usize) -> Result<(), E>, E>(
        &mut self,
        root: ExprRef,
        reserve: &F,
    ) -> Result<Vec<(ExprRef, ExprRef)>, PreparedSymbolicError<E>> {
        if self.failed || self.parts.is_none() || self.symbolic.is_none() {
            return Err(PreparedSymbolicError::failed());
        }
        self.failed = true;
        reserve(
            symbolic::operation_control_bytes::<F, E>()
                .ok_or_else(PreparedSymbolicError::overflow)?,
        )
        .map_err(PreparedSymbolicError::funding)?;
        let symbolic = self.symbolic.take().expect("checked symbolic source parts");
        let mut derivative = self.parts.take().expect("checked derivative source parts");
        let (symbolic, result) =
            symbolic.execute(self.source.source_mut(), &mut derivative, root, reserve);
        self.parts = Some(derivative);
        self.symbolic = Some(symbolic);
        if result.is_ok() {
            self.failed = false;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{raw::DerivCache, AlphabetInfo};
    #[test]
    fn owning_expression_machine_keeps_successor_memo_and_terminal_source_custody() {
        let mut source = ExprSet::new(256);
        let root = source.mk_byte_literal(&[b'x'; 64]);
        let (_, mut source, _) = AlphabetInfo::from_exprset(source, &[root]);
        source.reserve(48);
        let mut ordinary = source.clone();
        let mut cache = DerivCache::new();
        let prepared = source.prepared_source_plan().unwrap().compile().unwrap();
        drop(source);
        let plan = PreparedExpressionPlan::prepare(prepared).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.required_bytes(),
            requirements.buffer_bytes() + requirements.control_bytes()
        );
        let mut machine = plan.compile().unwrap();
        assert!(PreparedExpressionMachine::operation_control_bytes().unwrap() > 0);
        let mut current = root;
        let mut successes = 0usize;
        let mut refused = false;
        for _ in 0..64 {
            let expected = cache.derivative(&mut ordinary, current, b'x');
            match machine.derivative(current, b'x') {
                Ok(next) => {
                    assert_eq!(next, expected);
                    assert_eq!(machine.source().cost(), ordinary.cost());
                    assert_eq!(machine.num_derivatives(), Some(cache.num_deriv));
                    let calls = machine.num_derivatives();
                    assert_eq!(machine.derivative(current, b'x').unwrap(), next);
                    assert_eq!(machine.num_derivatives(), calls);
                    successes += 1;
                    current = next;
                }
                Err(error) => {
                    assert!(matches!(
                        error.cause,
                        workspace::OperationError::Process(OperationError::Arena(
                            PreparedExprError::Encoding(_)
                        ))
                    ));
                    refused = true;
                    break;
                }
            }
        }
        assert!(successes > 0);
        assert!(refused);
        let entries = machine.source().len();
        let cost = machine.source().cost();
        let calls = machine.num_derivatives();
        assert!(machine.source().is_valid(current));
        assert!(matches!(
            machine.derivative(root, b'x').unwrap_err().cause,
            workspace::OperationError::Process(OperationError::Failed)
        ));
        assert_eq!(machine.source().len(), entries);
        assert_eq!(machine.source().cost(), cost);
        assert_eq!(machine.num_derivatives(), calls);
        let retained = machine.source().expr_to_string(current);
        assert!(!retained.is_empty());
    }
    #[test]
    fn paid_successor_workspaces_preserve_memo_and_retain_exact_growth_refusal() {
        use crate::raw::PreparedHashConsFunding;
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        };
        #[derive(Debug)]
        struct Refusal;
        impl fmt::Display for Refusal {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("reached workspace refusal")
            }
        }
        impl std::error::Error for Refusal {}
        struct Account {
            reject: AtomicBool,
            retired: Arc<AtomicBool>,
            bytes: AtomicUsize,
        }
        impl Drop for Account {
            fn drop(&mut self) {
                self.retired.store(true, Ordering::SeqCst);
            }
        }
        for refuse in [false, true] {
            let mut original = ExprSet::new(256);
            let root = original.mk_byte_literal(&[b'x'; 64]);
            let (_, original, _) = AlphabetInfo::from_exprset(original, &[root]);
            let mut ordinary = original.clone();
            let mut cache = DerivCache::new();
            let mut prepared = original.prepared_source_plan().unwrap().compile().unwrap();
            let initial_nodes = prepared.source().prepared_extents().unwrap().1;
            let retired = Arc::new(AtomicBool::new(false));
            let account = Arc::new(Account {
                reject: AtomicBool::new(false),
                retired: retired.clone(),
                bytes: AtomicUsize::new(0),
            });
            let owner = account.clone();
            let funding = PreparedHashConsFunding::prepare(move |bytes| {
                if owner.reject.load(Ordering::SeqCst) {
                    return Err(Refusal);
                }
                owner.bytes.fetch_add(bytes, Ordering::SeqCst);
                Ok(())
            })
            .unwrap();
            prepared.bind_backing_funding(funding).unwrap();
            let mut machine = PreparedExpressionPlan::prepare(prepared)
                .unwrap()
                .compile()
                .unwrap();
            let mut current = root;
            let mut escaped_error = None;
            let mut grew = false;
            let mut detached_copy = None;
            for _ in 0..64 {
                if refuse && machine.source().len() > initial_nodes {
                    account.reject.store(true, Ordering::SeqCst);
                }
                let expected = cache.derivative(&mut ordinary, current, b'x');
                match machine.derivative(current, b'x') {
                    Ok(next) => {
                        assert_eq!(next, expected);
                        assert_eq!(machine.source().cost(), ordinary.cost());
                        assert_eq!(machine.num_derivatives(), Some(cache.num_deriv));
                        if !refuse || machine.source().len() <= initial_nodes {
                            let before = machine.num_derivatives();
                            assert_eq!(machine.derivative(current, b'x').unwrap(), next);
                            assert_eq!(machine.num_derivatives(), before);
                        }
                        current = next;
                        grew |= machine.source().len() > initial_nodes;
                    }
                    Err(error) => {
                        assert!(refuse);
                        let mut cause: &dyn std::error::Error = &error;
                        while let Some(next) = cause.source() {
                            cause = next;
                        }
                        assert!(cause.is::<Refusal>());
                        assert!(machine.source().is_valid(current));
                        assert!(machine.derivative(current, b'x').is_err());
                        assert!(machine.copy_required_bytes::<Refusal>().is_none());
                        escaped_error = Some(error);
                        break;
                    }
                }
            }
            assert!(grew);
            assert_eq!(escaped_error.is_some(), refuse);
            assert!(account.bytes.load(Ordering::SeqCst) > 0);
            if !refuse {
                use std::cell::Cell;
                let reserve = |_| Ok::<_, Refusal>(());
                assert!(machine.source().is_nullable(current));
                let expected_symbolic = machine.symbolic_derivative(root, &reserve).unwrap();
                assert!(machine.is_non_empty(root, u64::MAX, &reserve).unwrap());
                // The shared conservative shortcut proves repeat-shaped tails,
                // not arbitrary language equality. Compare its actual ordinary
                // answer for this long literal and preserve the cached false.
                let expected_containment = crate::relevance::RelevanceCache::new()
                    .is_contained_in_prefixes(&mut ordinary, &mut cache, root, root, u64::MAX, true)
                    .unwrap();
                assert!(!expected_containment);
                assert_eq!(machine.is_contained_in_prefixes(root, root, u64::MAX, true, &reserve).unwrap(), expected_containment);
                let copy_calls = Cell::new(0usize);
                let copy_required = machine.copy_required_bytes::<Refusal>().unwrap();
                let copy_spent = Cell::new(0usize);
                let source_cost_before_quote = machine.source().cost();
                assert_eq!(machine.copy_required_bytes::<Refusal>(), Some(copy_required));
                assert_eq!(machine.source().cost(), source_cost_before_quote);
                let copy_retired = Arc::new(AtomicBool::new(false));
                let copy_account = Arc::new(Account {
                    reject: AtomicBool::new(false), retired: copy_retired.clone(), bytes: AtomicUsize::new(0),
                });
                let copy_backing = PreparedHashConsFunding::prepare(move |bytes| {
                    copy_account.bytes.fetch_add(bytes, Ordering::SeqCst);
                    Ok::<_, Refusal>(())
                }).unwrap();
                let mut copied = machine.try_copy(Some(copy_backing), &|bytes| {
                    copy_calls.set(copy_calls.get() + 1);
                    let next = copy_spent.get().checked_add(bytes).ok_or(Refusal)?;
                    if next > copy_required { return Err(Refusal); }
                    copy_spent.set(next);
                    Ok::<_, Refusal>(())
                }).unwrap();
                assert_eq!(copy_spent.get(), copy_required);
                let short_spent = Cell::new(0usize);
                let short = machine.try_copy(None, &|bytes| {
                    let next = short_spent.get().checked_add(bytes).ok_or(Refusal)?;
                    if next >= copy_required { return Err(Refusal); }
                    short_spent.set(next);
                    Ok(())
                }).unwrap_err();
                assert!(short.to_string().contains("reached workspace refusal"));
                assert!(short_spent.get() < copy_required);
                drop(short);
                assert_eq!(copied.source().len(), machine.source().len());
                assert_eq!(copied.source().cost(), machine.source().cost());
                assert_eq!(copied.num_derivatives(), machine.num_derivatives());
                assert_eq!(copied.num_symbolic_nodes(), machine.num_symbolic_nodes());
                assert_ne!(copied.parts.as_ref().unwrap().buffers.memo.as_ptr(), machine.parts.as_ref().unwrap().buffers.memo.as_ptr());
                let memo_count = copied.num_derivatives();
                assert_eq!(copied.derivative(root, b'x').unwrap(), machine.derivative(root, b'x').unwrap());
                assert_eq!(copied.num_derivatives(), memo_count);
                assert_eq!(copied.symbolic_derivative(root, &reserve).unwrap(), expected_symbolic);
                assert!(copied.is_non_empty(root, u64::MAX, &reserve).unwrap());
                assert_eq!(copied.is_contained_in_prefixes(root, root, u64::MAX, true, &reserve).unwrap(), expected_containment);
                for cutoff in [1, copy_calls.get() / 2, copy_calls.get()] {
                    let attempted = Cell::new(0usize);
                    let error = machine.try_copy(None, &|_| {
                        attempted.set(attempted.get() + 1);
                        if attempted.get() == cutoff { Err(Refusal) } else { Ok(()) }
                    }).unwrap_err();
                    assert_eq!(attempted.get(), cutoff);
                    assert!(error.to_string().contains("reached workspace refusal"));
                    assert_eq!(machine.num_derivatives(), memo_count);
                }
                detached_copy = Some((copied, copy_retired));
            }
            drop(account);
            assert!(!retired.load(Ordering::SeqCst));
            drop(machine);
            if refuse {
                assert!(!retired.load(Ordering::SeqCst));
            }
            drop(escaped_error);
            assert!(retired.load(Ordering::SeqCst));
            if let Some((copied, copy_retired)) = detached_copy {
                // The copied source keeps only its explicit new growth account.
                assert!(!copy_retired.load(Ordering::SeqCst));
                assert!(copied.source().is_nullable(current));
                drop(copied);
                assert!(copy_retired.load(Ordering::SeqCst));
            }
        }
    }
}
