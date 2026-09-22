//! Ordinary allocation allowances from the exact selected group workers.
use super::{Group, GroupCpuStorageFacts};
use crate::{
    Array, Dtype, OperationEvalTraversalLimits, OperationEvent, OrdinaryControlPopulation, Stream,
};
use std::mem::{size_of, size_of_val};

/// Descriptive ordinary caller and native allocation allowances. Tensor
/// payload, persistent communicator storage and execution authority are separate.
#[derive(Clone, Copy, Debug)]
pub struct OrdinaryGroupControls {
    caller_bytes: usize,
    native: OrdinaryControlPopulation,
    primitives: usize,
    arrays: usize,
    edges: usize,
    rank: usize,
    captures: usize,
    extents: usize,
    completion_roots: usize,
    completions: usize,
}
impl OrdinaryGroupControls {
    /// Conservative caller-frame allowance from the selected producer and its
    /// descriptive layout query. Query transports remain part of the allowance;
    /// this is not an exact census of the ordinary execution stack.
    pub fn caller_control_bytes(self) -> usize {
        self.caller_bytes
    }
    /// Conservative physical-observer allowance for the selected allocation sources.
    pub fn allocation_controls(self) -> OrdinaryControlPopulation {
        self.native
    }
    /// Raw reachable graph nodes, array nodes and input edges. An enclosing
    /// ordinary equation combines these before pricing any shared Eval.
    pub fn graph_population(self) -> (usize, usize, usize) {
        (self.primitives, self.arrays, self.edges)
    }
    /// Maximum selected constructor rank.
    pub fn maximum_rank(self) -> usize {
        self.rank
    }
    /// Maximum selected CPU worker retained Data population.
    pub fn captures(self) -> usize {
        self.captures
    }
    /// Qualified constructor/dispatch request extents, excluding Eval.
    pub fn graph_extents(self) -> usize {
        self.extents
    }
    /// Completed stages and their conservative maximum root population.
    pub fn completion_frontier(self) -> (usize, usize) {
        (self.completion_roots, self.completions)
    }
    /// Sequential source composition; every counter remains checked.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            caller_bytes: self.caller_bytes.checked_add(other.caller_bytes)?,
            native: self.native.checked_add(other.native)?,
            primitives: self.primitives.checked_add(other.primitives)?,
            arrays: self.arrays.checked_add(other.arrays)?,
            edges: self.edges.checked_add(other.edges)?,
            rank: self.rank.max(other.rank),
            captures: self.captures.max(other.captures),
            extents: self.extents.checked_add(other.extents)?,
            completion_roots: self.completion_roots.max(other.completion_roots),
            completions: self.completions.checked_add(other.completions)?,
        })
    }
    pub(super) fn completion(mut self, limits: OperationEvalTraversalLimits) -> Option<Self> {
        // The enclosing equation owns Eval pricing. A logical exchange can
        // share its lazy graph with numerical dependency arithmetic, so these
        // sources retain a frontier rather than an isolated Eval allowance.
        self.completions = self.completions.checked_add(1)?;
        self.completion_roots = self.completion_roots.max(limits.roots);
        self.captures = self.captures.max(limits.captures);
        Some(self)
    }
}

/// The native query qualifies the same lazy constructor, copy and transport
/// workers used by ordinary calls. Graph extents describe individual allocator
/// requests, never a finished Original arena balance. The shared observer
/// mapping conservatively includes per-allocation headers and transports.
pub(super) fn leaf(
    constructor: &safemlx_sys::mlx_distributed_constructor_storage,
    evaluation: &GroupCpuStorageFacts,
) -> Option<OrdinaryGroupControls> {
    let (copy, worker) = evaluation.worker_graph_extents();
    let extents = constructor
        .allocation_extents
        .checked_add(evaluation.host_graph_extent())?
        .checked_add(copy)?
        .checked_add(worker)?;
    let frames = [
        crate::ops::ordinary_array_result_guard_control_bytes()?,
        Array::ordinary_clone_control_bytes()?,
        size_of::<(&Array, &Group, &Stream)>(),
        size_of::<(&[i32], Dtype, usize, &Group, &Stream)>(),
        size_of::<(&Array, usize, &Group, &Stream)>(),
        size_of::<(&Group, usize, &str)>(),
        size_of::<Option<crate::utils::runtime_lock::RuntimeLockGuard>>(),
        size_of::<Result<Array, crate::error::Exception>>(),
        size_of::<Result<i32, crate::error::Exception>>(),
        size_of::<Result<(), super::TerminalGroup>>(),
        size_of::<std::slice::Iter<'static, i32>>(),
        size_of::<(usize, usize, i32, Dtype)>(),
    ];
    let caller_bytes = frames.into_iter().try_fold(
        constructor
            .named_control_bytes
            .checked_add(evaluation.control_bytes()?)?
            .checked_add(size_of_val(&frames))?,
        usize::checked_add,
    )?;
    Some(OrdinaryGroupControls {
        caller_bytes,
        native: OperationEvent::ordinary_dispatch_control_envelope(extents)?,
        primitives: constructor.primitives,
        arrays: constructor
            .primitives
            .checked_add(constructor.input_edges)?
            .checked_add(1)?,
        edges: constructor.input_edges,
        rank: constructor.output_rank,
        captures: evaluation.retention_population().0,
        extents,
        completion_roots: 0,
        completions: 0,
    })
}

pub(super) fn variable_call_bytes(peers: usize) -> Option<usize> {
    // validate_all_to_all_v owns both converted count vectors and the offsets
    // vector. The matrix and tensor data stay with their existing source owners.
    let cells = peers.checked_mul(
        size_of::<i64>()
            .checked_mul(2)?
            .checked_add(size_of::<usize>())?,
    )?;
    let frames = [
        size_of::<(&Array, &[usize], &[usize], &Group, &Stream)>(),
        size_of::<(&Array, &[i64], &[i64], &Group, &Stream)>(),
        size_of::<(Vec<i64>, Vec<i64>, Vec<usize>)>(),
        size_of::<Result<(Vec<i64>, Vec<i64>, Vec<usize>), crate::error::Exception>>(),
        size_of::<std::slice::Iter<'static, usize>>().checked_mul(3)?,
        size_of::<usize>().checked_mul(8)?,
        cells,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

impl OrdinaryGroupControls {
    pub(super) fn with_variable_call(mut self, peers: usize) -> Option<Self> {
        self.caller_bytes = self.caller_bytes.checked_add(variable_call_bytes(peers)?)?;
        Some(self)
    }
}
