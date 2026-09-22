//! Ordinary CPU copies use the retained Host geometry and the shared native
//! constructors, Eval traversal and dispatch sources. These are prospective
//! allowances, separate from payload, C wrappers and manager metadata.
use super::*;
use crate::backend::nn::workspace::OrdinaryNativeControls;
use safemlx::{DeviceType, Dtype, OperationEvalTraversalLimits, OperationEvent, StreamCopyPlan};

fn completion(roots: usize, copy: bool) -> Option<OrdinaryNativeControls> {
    let tape = if copy { 2 } else { 1 };
    let record = OperationEvent::eval_record_layout(tape, 1, tape)?;
    let limits = OperationEvalTraversalLimits {
        roots,
        arrays: roots.checked_add(tape)?,
        tape_entries: tape,
        input_edges: roots.checked_add(usize::from(copy))?,
        output_slots: tape,
        streams: 1,
        captures: record.capture_slots().max(1),
    };
    let mut controls = OrdinaryNativeControls::default();
    controls.include(OperationEvent::ordinary_cpu_eval_control_layout(limits)?)?;
    controls.include(OperationEvent::ordinary_array_vector_control_layout(roots)?)?;
    let worker = OperationEvent::cpu_completion_layout(roots)?;
    controls.include(OperationEvent::ordinary_cpu_dispatch_envelope(
        worker
            .graph_allocation_extents()
            .checked_add(worker.worker_graph_allocation_extents())?
            .checked_add(worker.signal_graph_allocation_extents())?,
    )?)?;
    Some(controls)
}

fn copy(dtype: Dtype, rank: usize, store: bool) -> Option<OrdinaryNativeControls> {
    let mut controls = completion(1, true)?;
    // Four is the shared constructor's actual minimum operand-vector capacity;
    // the Eval traversal still contains exactly one copy input edge.
    controls.include(OperationEvent::ordinary_frontend_control_layout(
        1,
        usize::from(!store),
        rank,
        4,
    )?)?;
    let worker = OperationEvent::cpu_host_transfer_layout(dtype, rank, store, false)?;
    controls.include(OperationEvent::ordinary_cpu_dispatch_envelope(
        worker
            .graph_allocation_extents()
            .checked_add(worker.worker_graph_allocation_extents())?
            .checked_add(worker.signal_graph_allocation_extents())?,
    )?)?;
    Some(controls)
}

impl ResidencyManager {
    /// One ordinary transfer of a completed contiguous input, including its
    /// actual completion. Host construction, physical backing and C wrappers
    /// remain separate. The store route consumes an existing Array; the load
    /// route constructs one actual seed descriptor over retained Host storage.
    pub(in crate::backend::runtime::residency::manager) fn ordinary_host_transfer_controls(
        dtype: Dtype,
        rank: usize,
        store: bool,
    ) -> Option<OrdinaryNativeControls> {
        copy(dtype, rank, store)
    }
    pub(crate) fn ordinary_transfer_inspection_control_bytes() -> Option<usize> {
        let frames = [
            StreamCopyPlan::<()>::capture_control_bytes().ok()?,
            size_of::<StreamCopyPlan<()>>(),
            size_of::<Result<StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<OrdinaryNativeControls>() * 3,
            size_of::<Option<OrdinaryNativeControls>>(),
            size_of::<Result<Option<OrdinaryNativeControls>, ResidencyError>>(),
            size_of::<OperationEvalTraversalLimits>(),
            size_of::<safemlx::OperationEvalRecordLayout>(),
            size_of::<safemlx::CpuCopyEvalLayout>(),
            size_of::<safemlx::OrdinaryControlPopulation>(),
            size_of::<(
                &Self,
                &[OffloadUnitId],
                &mut [ResidencyClosureSlot],
                WindowPopulation,
            )>(),
            size_of::<(&OffloadUnit, &WeightBinding)>(),
            size_of::<Option<(&[i32], Dtype)>>(),
            size_of::<(usize, bool, Dtype, usize)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Every eligible canonical row must have an authentic retained Host source.
    /// Per-row maxima compose with the selected window's physical copy count;
    /// named aliases extend only the final submitted-root population.
    pub(crate) fn ordinary_transfer_controls(
        &self,
        roots: &[OffloadUnitId],
        scratch: &mut [ResidencyClosureSlot],
        window: WindowPopulation,
    ) -> Result<Option<OrdinaryNativeControls>, ResidencyError> {
        let state = self.inner.state.try_lock().map_err(|cause| match cause {
            std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
            std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
        })?;
        let stream = StreamCopyPlan::<()>::capture(&state.device_stream)
            .map_err(|_| ResidencyError::OrdinaryHostControlSource)?;
        if stream.device_type() != DeviceType::Cpu {
            return Ok(None);
        }
        let closure = state
            .control
            .operation_closure(roots, scratch)
            .map_err(ResidencyError::OperationClosure)?;
        let mut per_copy = OrdinaryNativeControls::default();
        for unit in closure.units() {
            for binding in unit.bindings().iter().filter(|binding| !binding.is_alias()) {
                let geometry = match self.inner.sources.prepared_host(unit.id()) {
                    Some(host) => host
                        .buffers
                        .get(binding.name())
                        .and_then(|buffer| buffer.prepared_metadata())
                        .map(|metadata| (metadata.shape(), metadata.dtype())),
                    None => self
                        .inner
                        .sources
                        .foreground()
                        .and_then(|source| source.native_output(unit.id(), binding.name())),
                };
                let Some((shape, dtype)) = geometry else {
                    return Ok(None);
                };
                let Some(controls) = copy(dtype, shape.len(), false) else {
                    return Ok(None);
                };
                per_copy = per_copy.union(controls);
            }
        }
        let Some(mut aggregate) = completion(window.bindings, false) else {
            return Ok(None);
        };
        let Some(wait) = OperationEvent::ordinary_cpu_wait_control_layout() else {
            return Ok(None);
        };
        let overflow = || ResidencyError::ArithmeticOverflow {
            context: "ordinary transfer control population",
        };
        aggregate.include(wait).ok_or_else(overflow)?;
        per_copy
            .repeat(window.physical_bindings)
            .and_then(|copies| copies.append(aggregate))
            .map(Some)
            .ok_or_else(overflow)
    }
}
