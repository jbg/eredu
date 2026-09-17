//! Actual accepted Ring primitive/Eval source; no physical backing or group fit.
use super::{GroupStorageUnavailable,GroupWorkerOperation};
use crate::Array;
use std::{fmt,mem::{size_of,size_of_val}};

/// A borrowed native output keeps its exact primitive, group and input graph.
/// Host Graph requests, asynchronous workers and logical backing sizes remain
/// separate; this descriptor cannot manufacture Scope or buffer authority.
pub struct GroupCpuEvaluationStorage<'a> {
    output:&'a Array,
    operation:GroupWorkerOperation,
    facts:GroupCpuStorageFacts,
}
/// Descriptive allocation populations emitted by the one native CPU worker.
/// These facts carry no allocation or submission authority by themselves.
#[derive(Debug)]
pub struct GroupCpuStorageFacts {
    value:safemlx_sys::mlx_distributed_cpu_eval_storage,
}
impl fmt::Debug for GroupCpuEvaluationStorage<'_> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("GroupCpuEvaluationStorage").field("operation",&self.facts.value.operation)
            .field("blocks",&self.facts.value.blocks).finish_non_exhaustive()
    }
}
impl Array {
    /// Pure current-source query for the accepted native CPU Ring primitive.
    /// Unknown implementations, unrelated/subclass primitives and malformed
    /// geometry refuse. Canonical fencing does not invalidate accepted graphs.
    pub fn distributed_cpu_evaluation_storage(&self)
        ->Result<GroupCpuEvaluationStorage<'_>,GroupStorageUnavailable> {
        let mut value=safemlx_sys::mlx_distributed_cpu_eval_storage::default();
        // SAFETY: immutable retained array/primitive source, metadata-only C query.
        if !unsafe{safemlx_sys::mlx_distributed_query_cpu_eval_storage(&mut value,self.as_ptr())} {
            return Err(GroupStorageUnavailable);
        }
        let operation=match value.operation {
            0=>GroupWorkerOperation::Sum,1=>GroupWorkerOperation::Maximum,
            2=>GroupWorkerOperation::Minimum,3=>GroupWorkerOperation::Gather,
            4=>GroupWorkerOperation::Send{peer:value.peer},
            5=>GroupWorkerOperation::Receive{peer:value.peer},
            6=>GroupWorkerOperation::Variable,
            _=>return Err(GroupStorageUnavailable),
        };
        Ok(GroupCpuEvaluationStorage{output:self,operation,facts:GroupCpuStorageFacts{value}})
    }
    /// Source-dependent fixed inspection frames, payable before the pure query.
    pub fn distributed_cpu_evaluation_control_bytes(&self)->Option<usize> {
        // SAFETY: reads fixed layout and actual retained primitive/source metadata only.
        let native=unsafe{safemlx_sys::mlx_distributed_cpu_eval_storage_controls(self.as_ptr())};
        let controls=[size_of::<GroupCpuEvaluationStorage<'_>>(),size_of::<&Self>(),
            size_of::<Result<GroupCpuEvaluationStorage<'_>,GroupStorageUnavailable>>(),
            size_of::<GroupStorageUnavailable>(),size_of::<bool>()];
        controls.into_iter().try_fold(native.checked_add(size_of_val(&controls))?,usize::checked_add)
    }
}
impl GroupCpuEvaluationStorage<'_> {
    /// Exact primitive operation and native peer, derived from the retained graph.
    pub fn operation(&self)->GroupWorkerOperation{self.operation}
    /// Exact output-wrapper identity; the borrowed array retains actual sources.
    pub fn is_for(&self,output:&Array)->bool{std::ptr::eq(self.output,output)}
    /// The same closed facts used by the cold source; output identity remains
    /// carried by this accepted-source loan.
    pub fn facts(&self)->&GroupCpuStorageFacts { &self.facts }
    /// Actual input and output ranks and input-edge population.
    pub fn geometry(&self)->(usize,usize,usize){self.facts.geometry()}
    /// Source-derived host bank requests (bytes, alignment, attempts).
    pub fn classes(&self)->impl ExactSizeIterator<Item=(usize,usize,usize)>+'_ {self.facts.classes()}
    /// Graph extent of the actual host bank.
    pub fn host_graph_extent(&self)->usize{self.facts.host_graph_extent()}
    /// Copy-scratch and Ring worker extents.
    pub fn worker_graph_extents(&self)->(usize,usize){self.facts.worker_graph_extents()}
    /// Actual paths' maximum backing births and logical bytes.
    pub fn logical_backing_population(&self)->(usize,usize){self.facts.logical_backing_population()}
    /// Potential copy and output logical allocation sizes.
    pub fn logical_backing_requests(&self)->(usize,usize){self.facts.logical_backing_requests()}
    /// Data wrapper and temporary-batch population.
    pub fn retention_population(&self)->(usize,usize){self.facts.retention_population()}
    /// Complete named source/fact transport controls.
    pub fn control_bytes(&self)->Option<usize>{self.facts.control_bytes()?.checked_add(size_of::<Self>())}
}
impl GroupCpuStorageFacts {
    pub(super) fn native(&self)->&safemlx_sys::mlx_distributed_cpu_eval_storage {&self.value}
    pub(super) fn from_native(value:safemlx_sys::mlx_distributed_cpu_eval_storage)->Self{Self{value}}
    /// Actual input and output ranks and input-edge population.
    pub fn geometry(&self)->(usize,usize,usize){(self.value.input_rank,self.value.output_rank,self.value.inputs)}
    /// Source-derived host bank requests (bytes, alignment, attempts).
    pub fn classes(&self)->impl ExactSizeIterator<Item=(usize,usize,usize)>+'_ {
        self.value.request_bytes.iter().copied().zip(self.value.request_alignments.iter().copied())
            .zip(self.value.request_counts.iter().copied()).map(|((bytes,alignment),count)|(bytes,alignment,count))
    }
    /// Host Graph extent including bank header/slots; successful actual preflight
    /// is still required. No fragmented-space or physical-buffer promise.
    pub fn host_graph_extent(&self)->usize{self.value.allocation_extents}
    /// Separate copy-scratch and Ring task/plan destination extents.
    pub fn worker_graph_extents(&self)->(usize,usize){(self.value.copy_worker_graph_extent,self.value.communication_worker_graph_extent)}
    /// Maximum backing births and total logical bytes along the actual paths.
    /// Convert each actual allocation through the selected native buffer facts;
    /// these logical bytes are not physical capacity or an allocation grant.
    pub fn logical_backing_population(&self)->(usize,usize){(self.value.backing_births,self.value.logical_backing_bytes)}
    /// Potential copy and output logical allocation sizes. Reduction uses at
    /// most one of them; gather uses both; send uses copy; receive uses output.
    pub fn logical_backing_requests(&self)->(usize,usize){(self.value.copy_backing_bytes,self.value.output_backing_bytes)}
    /// New Data wrappers (including weak copy captures) and temporary batches.
    pub fn retention_population(&self)->(usize,usize){(self.value.data_captures,self.value.temporary_batches)}
    /// Native host and worker frame census, separate from allocation extents.
    pub fn control_bytes(&self)->Option<usize> {
        self.value.named_control_bytes.checked_add(size_of::<Self>())?.checked_add(size_of::<&Self>())
    }
}
