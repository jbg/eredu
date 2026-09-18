//! Source-derived host materialization populations, independent of neural Eval.
use super::*;
use crate::backend::error::Error;
use crate::backend::runtime::residency::manager::{
    ForegroundDiskDescriptors, ForegroundDiskPopulation, HostCopyWorkspace,
};
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{DeviceType, ImmutableHostTransferBuffer, OperationEvent};
use super::super::cpu::CpuPopulation;

/// The actual consumer either constructs fresh units or rebinds retained slots.
/// An unbound source copy recipe cannot establish either producer.
#[derive(Clone, Copy, Debug)]
pub(super) enum HostParameterConstruction {
    FreshUnits,
    RetainedModules,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct HostCopies {
    pub(super) parameters: Option<HostParameterConstruction>,
    pub(super) per_forward: usize,
    pub(super) traversal: safemlx::OperationEvalTraversalLayout,
    pub(super) dispatch: ResidentDispatchPopulation,
    pub(super) direct_graph_extents: usize,
    pub(super) aggregate_traversal: safemlx::OperationEvalTraversalLayout,
    pub(super) aggregate_dispatch: ResidentDispatchPopulation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostTransfers {
    pub(super) per_forward: usize,
    pub(super) waits_per_forward: usize,
    pub(super) roots: usize,
    attempts_per_window: usize,
    pub(super) copies: usize,
    pub(super) binding_shells: usize,
    pub(super) bytes: u64,
}
/// Exact immutable source and accepted slot populations retained by the recipe.
/// Native copy/aggregate/wait populations stay in the shared HostCopies fields.
pub(super) struct ForegroundCopies {
    device: DeviceType,
    source: ForegroundDiskDescriptors,
    per_forward: ForegroundDiskPopulation,
    request: ForegroundDiskPopulation,
}
impl std::fmt::Debug for ForegroundCopies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForegroundCopies")
            .field("device", &self.device)
            .field("per_forward", &self.per_forward)
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}
#[derive(Default)]
struct CopyLayout {
    rank: usize,
    controls: usize,
    direct_extent: usize,
    cpu: CpuPopulation,
}
impl CopyLayout {
    fn include(&mut self, rank: usize, dtype: safemlx::Dtype, device: DeviceType) -> Option<()> {
        let (controls, direct) = ImmutableHostTransferBuffer::original_copy_layout(rank, dtype)?;
        self.rank = self.rank.max(rank);
        self.controls = self.controls.max(controls);
        self.direct_extent = self.direct_extent.max(direct);
        if device == DeviceType::Cpu {
            let mut population = CpuPopulation::default();
            population.copy(OperationEvent::cpu_host_transfer_layout(dtype, rank, false, false)?, 1)?;
            // Each independently completed copy uses the largest actual source
            // row's bank. Every scalar field bounds those same finite rows; no
            // unsupported row is discarded and no source backing is a birth.
            self.cpu.construction_entries = self.cpu.construction_entries.max(population.construction_entries);
            self.cpu.primitives = self.cpu.primitives.max(population.primitives);
            self.cpu.input_edges = self.cpu.input_edges.max(population.input_edges);
            self.cpu.maximum_operands = self.cpu.maximum_operands.max(population.maximum_operands);
            self.cpu.maximum_captures = self.cpu.maximum_captures.max(population.maximum_captures);
            self.cpu.births = self.cpu.births.max(population.births);
            self.cpu.extents = self.cpu.extents.max(population.extents);
            self.cpu.controls = self.cpu.controls.max(population.controls);
        }
        Some(())
    }
}
#[derive(Clone, Copy)]
pub(super) struct PreparedSourceCopies {
    device: DeviceType,
    pub(super) copies: HostCopies,
    pub(super) transfers: HostTransfers,
    pub(super) rank: usize,
    pub(super) controls: usize,
}
impl PreparedSourceCopies {
    pub(super) fn inspection_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [size_of::<Self>(), size_of::<Result<Self,Error>>(),
            size_of::<CopyLayout>(), size_of::<HostTransfers>() * 2,
            size_of::<crate::backend::runtime::residency::manager::WindowPopulation>(),
            size_of::<(&crate::backend::runtime::residency::manager::SupplementaryResidencySource,usize)>(),
            size_of::<(usize,usize,usize,usize)>(),
            size_of::<Result<(usize,usize,usize,usize),Error>>(),
            size_of::<[usize;5]>(), size_of::<HostCopies>(), size_of::<DeviceType>(),
            size_of::<CpuPopulation>(), size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<safemlx::CpuCopyEvalLayout>(),
            size_of::<(usize,safemlx::Dtype,DeviceType,&mut CopyLayout)>(),
            size_of::<Result<(ResidentDispatchPopulation,usize),Error>>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    fn prepare(layout: CopyLayout, transfers: HostTransfers, additional_controls: usize, device: DeviceType)
        -> Result<Self, Error> {
        let error = Error::PrefillControl;
        let unknown = || error(WorkingMemoryError::UnknownBound);
        let overflow = || error(WorkingMemoryError::Overflow);
        let count = transfers.copies;
        let CopyLayout {
            rank,
            controls: per_copy_controls,
            direct_extent, cpu,
        } = layout;
        if count != 0 && (direct_extent == 0 || per_copy_controls == 0) {
            return Err(unknown());
        }
        let per_copy_controls = per_copy_controls
            .checked_add(
                crate::backend::runtime::residency::manager::original_host_copy_control_bytes()
                    .ok_or_else(unknown)?,
            )
            .ok_or_else(overflow)?;
        let mut controls = per_copy_controls
            .checked_mul(count)
            .and_then(|bytes| bytes.checked_add(additional_controls))
            .ok_or_else(overflow)?;
        let direct_graph_extents = direct_extent.checked_mul(count).ok_or_else(overflow)?;
        // The actual copy Eval contains its host leaf, CopyFromHostTransfer and
        // Synchronizer. Its one stream is the retained admitted execution stream.
        let record = OperationEvent::eval_record_layout(2, 1, 2).ok_or_else(unknown)?;
        let traversal =
            OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots: 1,
                arrays: 3,
                tape_entries: 2,
                input_edges: 2,
                output_slots: 2,
                streams: 1,
                captures: record.capture_slots().max(1),
            })
            .ok_or_else(unknown)?;
        let (dispatch, worker_controls) = copy_dispatch(device, traversal, rank, cpu)
            .ok_or_else(unknown)?;
        let aggregate_record = OperationEvent::eval_record_layout(1, 1, 1).ok_or_else(unknown)?;
        let aggregate_traversal =
            OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots: transfers.roots,
                arrays: transfers.roots.checked_add(1).ok_or_else(overflow)?,
                tape_entries: 1,
                input_edges: transfers.roots,
                output_slots: 1,
                streams: 1,
                captures: aggregate_record.capture_slots().max(1),
            })
            .ok_or_else(unknown)?;
        let (aggregate_dispatch, aggregate_worker_controls) =
            copy_dispatch(device, aggregate_traversal, rank, CpuPopulation::default())
                .ok_or_else(unknown)?;
        controls = controls
            .checked_add(OperationEvent::nested_scheduled_control_bytes().ok_or_else(unknown)?)
            .and_then(|n| n.checked_add(aggregate_record.query_control_bytes()?))
            .and_then(|n| n.checked_add(aggregate_traversal.query_control_bytes()?))
            .and_then(|n| n.checked_add(aggregate_worker_controls))
            .ok_or_else(overflow)?;
        controls = controls
            .checked_add(record.query_control_bytes().ok_or_else(unknown)?)
            .and_then(|n| n.checked_add(traversal.query_control_bytes()?))
            .and_then(|n| n.checked_add(worker_controls))
            .and_then(|n| n.checked_add(std::mem::size_of::<HostCopies>()))
            .ok_or_else(overflow)?;
        controls = controls.checked_add(Self::inspection_control_bytes().ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        Ok(Self { device, copies: HostCopies { parameters: None, per_forward: count,
            traversal, dispatch, direct_graph_extents, aggregate_traversal, aggregate_dispatch },
            transfers, rank, controls })
    }
    pub(super) fn inspect_layerwise(
        source:&crate::backend::runtime::execution::generic::LayerwiseWorkspace,
        windows:&[crate::backend::runtime::residency::manager::WindowPopulation],
    )->Result<Self,Error> {
        let unknown=||Error::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow=||Error::PrefillControl(WorkingMemoryError::Overflow);
        let device=source.destination_device_type();
        let mut layout=CopyLayout::default();
        if !source.visit_native_copy_layouts(|rank,dtype|layout.include(rank,dtype,device).ok_or_else(unknown))? {
            return Err(unknown());
        }
        let mut transfers=0usize;let mut waits=0usize;let mut roots=0usize;
        let mut attempts=None;
        for window in windows.iter().filter(|window|window.requested!=0) {
            let (calls,observations,outputs,tries)=
                crate::backend::runtime::execution::generic::OriginalSelectedResidencyAttempt::native_transfer_population(*window)?;
            if attempts.is_some_and(|value|value!=tries) {return Err(unknown());}
            attempts=Some(tries);
            transfers=transfers.checked_add(calls).ok_or_else(overflow)?;
            waits=waits.checked_add(observations).ok_or_else(overflow)?;
            roots=roots.max(outputs);
        }
        let transfers=HostTransfers::from_windows(1,transfers,waits,roots,
            attempts.unwrap_or(0),windows).ok_or_else(overflow)?;
        Self::prepare(layout,transfers,0,device)
    }
    pub(super) fn inspect(
        source: &crate::backend::runtime::residency::manager::SupplementaryResidencySource,
        window: crate::backend::runtime::residency::manager::WindowPopulation,
        occurrences: usize,
    ) -> Result<Self, Error> {
        let error = Error::PrefillControl;
        let unknown = || error(WorkingMemoryError::UnknownBound);
        let overflow = || error(WorkingMemoryError::Overflow);
        if occurrences == 0 || window.requested == 0 { return Err(unknown()); }
        let mut layout = CopyLayout::default();
        let device = match (source.host(), source.foreground()) {
            (Some(host), None) => {
                let device = host.destination_device_type();
                for unit in host.units() { for copy in host.copies(unit) {
                    layout.include(copy.shape().len(), copy.dtype(), device).ok_or_else(unknown)?;
                } }
                device
            },
            (None, Some(disk)) => {
                let device = disk.destination_device_type();
                for (shape, dtype) in disk.source().native_reads() {
                    layout.include(shape.len(), dtype, device).ok_or_else(unknown)?;
                }
                device
            },
            _ => return Err(unknown()),
        };
        // One warm and one missing owner are the existing finite window worker.
        // Failed whole-unit retries may copy the physical closure twice.
        let (transfers, waits, roots, attempts) =
            crate::backend::runtime::execution::generic::OriginalSelectedResidencyAttempt::native_transfer_population(window)?;
        let per_window = HostTransfers::from_windows(1, transfers, waits,
            roots, attempts, &[window]).ok_or_else(overflow)?;
        let transfers = HostTransfers {
            per_forward: per_window.per_forward.checked_mul(occurrences).ok_or_else(overflow)?,
            waits_per_forward: per_window.waits_per_forward.checked_mul(occurrences).ok_or_else(overflow)?,
            roots: per_window.roots,
            attempts_per_window: per_window.attempts_per_window,
            copies: per_window.copies.checked_mul(occurrences).ok_or_else(overflow)?,
            binding_shells: per_window.binding_shells.checked_mul(occurrences).ok_or_else(overflow)?,
            bytes: per_window.bytes.checked_mul(u64::try_from(occurrences).map_err(|_|overflow())?).ok_or_else(overflow)?,
        };
        Self::prepare(layout, transfers, 0, device)
    }
}

fn copy_dispatch(device: DeviceType, traversal: safemlx::OperationEvalTraversalLayout,
    rank: usize, cpu: CpuPopulation) -> Option<(ResidentDispatchPopulation, usize)> {
    let limits = traversal.limits();
    let (cpu_model, gpu_entries, gpu_edges, gpu_siblings, gpu_births,
        cpu_entries, cpu_edges, cpu_siblings, extents, kernels, controls) = match device {
        DeviceType::Cpu => {
            if limits.streams != 1 || limits.tape_entries != cpu.primitives.checked_add(1)?
                || limits.input_edges != cpu.input_edges.checked_add(limits.roots)? {
                return None;
            }
            let completion = OperationEvent::cpu_completion_layout(limits.roots)?;
            if completion.backing_births() != 0 || completion.worker_graph_allocation_extents() != 0 {
                return None;
            }
            (Some(cpu), 0, 0, 0, 0, limits.tape_entries, limits.input_edges,
                limits.tape_entries, 0, 0, cpu.controls.checked_add(completion.control_bytes()?)?)
        }
        DeviceType::Gpu => {
            let births = limits.tape_entries.checked_sub(1)?;
            let worker = OperationEvent::resident_gpu_worker_layout(limits.tape_entries,
                limits.input_edges, limits.output_slots, limits.arrays, births, rank, 4)?;
            (None, limits.tape_entries, limits.input_edges, limits.output_slots, births,
                0, 0, 0, worker.allocation_extents(), worker.kernel_attempts(), worker.control_bytes()?)
        }
    };
    let dispatch = ResidentDispatchPopulation { cpu_model,
        gpu_entries, gpu_input_edges: gpu_edges, gpu_siblings, gpu_births,
        additional_sort_kernels: 0, cpu_entries, cpu_input_edges: cpu_edges, cpu_siblings,
        parallel_entries: 0, parallel_graph_extents: 0, worker_graph_extents: extents,
        worker_rank: rank, copy_rank_extents: 0, kernel_attempts: kernels };
    let frames = [controls, std::mem::size_of::<(DeviceType,safemlx::OperationEvalTraversalLayout,
        usize,CpuPopulation,safemlx::OperationEvalTraversalLimits,ResidentDispatchPopulation)>(),
        std::mem::size_of::<(Option<CpuPopulation>,usize,usize,usize,usize,usize,usize,usize,usize,usize,usize)>(),
        std::mem::size_of::<Option<safemlx::CpuCopyEvalLayout>>(),std::mem::size_of::<safemlx::CpuCopyEvalLayout>(),
        std::mem::size_of::<Option<safemlx::ResidentGpuWorkerLayout>>(),std::mem::size_of::<safemlx::ResidentGpuWorkerLayout>(),
        std::mem::size_of::<Option<(ResidentDispatchPopulation,usize)>>()];
    Some((dispatch, frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)?))
}
impl PreparedSourceCopies {
    pub(super) fn expand_equation_dispatch(&self, mut dispatch: ResidentDispatchPopulation,
        limits: safemlx::OperationEvalTraversalLimits, operands: usize)
        -> Result<(ResidentDispatchPopulation,usize),Error> {
        let unknown = || Error::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow = || Error::PrefillControl(WorkingMemoryError::Overflow);
        if (self.device == DeviceType::Cpu) != dispatch.cpu_model.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        if dispatch.completion_streams() != Some(limits.streams) { return Err(unknown()); }
        dispatch.worker_rank = dispatch.worker_rank.max(self.rank);
        let worker_controls = match self.device {
            DeviceType::Cpu => {
                // CPU Eval population belongs to the unchanged equations. The
                // additional completed arrays require traversal slots only.
                0
            }
            DeviceType::Gpu => {
                let worker = OperationEvent::resident_gpu_worker_layout_with_router(
                    dispatch.gpu_entries, dispatch.gpu_input_edges, dispatch.gpu_siblings,
                    limits.arrays, dispatch.gpu_births, dispatch.worker_rank, operands,
                    dispatch.additional_sort_kernels,
                    dispatch.cpu_entries.checked_sub(dispatch.parallel_entries).ok_or_else(unknown)?)
                    .ok_or_else(unknown)?;
                dispatch.worker_graph_extents = worker.allocation_extents()
                    .checked_add(dispatch.copy_rank_extents)
                    .and_then(|n|n.checked_add(dispatch.parallel_graph_extents)).ok_or_else(overflow)?;
                dispatch.kernel_attempts = worker.kernel_attempts();
                worker.control_bytes().ok_or_else(unknown)?
            }
        };
        let frames = [worker_controls, ResidentDispatchPopulation::completion_stream_control_bytes(),
            std::mem::size_of::<(&Self,ResidentDispatchPopulation,
            safemlx::OperationEvalTraversalLimits,usize)>(),
            std::mem::size_of::<safemlx::ResidentGpuWorkerLayout>(),
            std::mem::size_of::<Option<safemlx::ResidentGpuWorkerLayout>>(),
            std::mem::size_of::<Result<(ResidentDispatchPopulation,usize),Error>>()];
        let controls = frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
            .ok_or_else(overflow)?;
        Ok((dispatch,controls))
    }
}

impl ResidentNativeRecipe {
    /// The retained source snapshot supplies actual canonical copy rows. The
    /// same lookahead windows and finite warm/missing attempt slots bound every
    /// copy, including failed prefixes. Cache hits only remove calls; no early
    /// output retirement or source donation credit is used.
    pub(crate) fn bind_host_copies(&mut self, source: &HostCopyWorkspace) -> Result<(), Error> {
        if self.host_copies.is_some() || self.neural.is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut layout = CopyLayout::default();
        for unit in source.units() {
            for copy in source.copies(unit) {
                layout
                    .include(copy.shape().len(), copy.dtype(), source.destination_device_type())
                    .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            }
        }
        self.bind_source_copies(layout, 0, source.destination_device_type())
    }

    /// The actual disk slot plan supplies these populations. Canonical windows
    /// still bound all copies, including initial immutable host sources that
    /// later miss on Device; a disk-only subset must not replace that bound.
    pub(crate) fn bind_foreground_disk_copies(
        &mut self,
        source: &ForegroundDiskDescriptors,
        per_forward: ForegroundDiskPopulation,
        request: ForegroundDiskPopulation,
        device: DeviceType,
    ) -> Result<(), Error> {
        let error = Error::PrefillControl;
        let unknown = || error(WorkingMemoryError::UnknownBound);
        let identity = || error(WorkingMemoryError::IdentityMismatch);
        let overflow = || error(WorkingMemoryError::Overflow);
        if self.foreground_copies.is_some() || self.host_copies.is_some() {
            return Err(identity());
        }
        let forwards = self.plan.generation_forward_count().ok_or_else(unknown)?;
        let transfers = self.host_transfers.ok_or_else(unknown)?;
        if per_forward.checked_mul(forwards).ok_or_else(overflow)? != request
            || per_forward.outputs > transfers.copies
            || u64::try_from(per_forward.output_logical_bytes).map_err(|_| overflow())?
                > transfers.bytes
        {
            return Err(identity());
        }
        let mut layout = CopyLayout::default();
        for (shape, dtype) in source.native_reads() {
            layout.include(shape.len(), dtype, device).ok_or_else(unknown)?;
        }
        if per_forward.maximum_rank > layout.rank {
            return Err(identity());
        }
        // The row already owns its inline recipe; these are the actual borrowed
        // query and retained-handle transports. Source backing was accepted by
        // the read-slot bank and is never added to device mutable bytes here.
        let controls = [
            std::mem::size_of::<CopyLayout>(),
            std::mem::size_of::<ForegroundCopies>(),
            std::mem::size_of::<ForegroundDiskDescriptors>(),
            std::mem::size_of::<ForegroundDiskPopulation>(),
            std::mem::size_of::<Option<ForegroundDiskPopulation>>(),
            std::mem::size_of::<&ForegroundDiskDescriptors>(),
            std::mem::size_of::<(DeviceType, &Self, &ForegroundDiskDescriptors,
                ForegroundDiskPopulation, ForegroundDiskPopulation, bool)>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(overflow)?;
        self.bind_source_copies(layout, controls, device)?;
        self.foreground_copies = Some(ForegroundCopies {
            device,
            source: source.clone(),
            per_forward,
            request,
        });
        Ok(())
    }

    pub(crate) fn matches_foreground_disk_copies(
        &self,
        source: &ForegroundDiskDescriptors,
        per_forward: ForegroundDiskPopulation,
        request: ForegroundDiskPopulation,
        device: DeviceType,
    ) -> bool {
        self.host_copies.is_some()
            && self.foreground_copies.as_ref().is_some_and(|copies| {
                copies.device == device && copies.source.same_source(source)
                    && copies.per_forward == per_forward
                    && copies.request == request
                    && self
                        .plan
                        .generation_forward_count()
                        .and_then(|forwards| per_forward.checked_mul(forwards))
                        == Some(request)
            })
    }

    pub(crate) fn matches_foreground_disk_source(
        &self,
        source: &ForegroundDiskDescriptors,
        device: DeviceType,
    ) -> bool {
        self.host_copies.is_some()
            && self
                .foreground_copies
                .as_ref()
                .is_some_and(|copies| copies.device == device && copies.source.same_source(source))
    }

    pub(crate) fn has_foreground_disk_copy_recipe(&self) -> bool {
        self.foreground_copies.is_some()
    }

    /// Each accepted disk output may add one explicit immutable host owner to
    /// the same device publication. Duplicate aliases are deduplicated by the
    /// collector; counting every attempted source covers retained failure rows.
    pub(crate) fn foreground_source_publication_rows(&self) -> usize {
        self.foreground_copies
            .as_ref()
            .map_or(0, |copies| copies.request.outputs)
    }

    fn bind_source_copies(
        &mut self,
        layout: CopyLayout,
        additional_controls: usize,
        device: DeviceType,
    ) -> Result<(), Error> {
        let error = Error::PrefillControl;
        let unknown = || error(WorkingMemoryError::UnknownBound);
        let overflow = || error(WorkingMemoryError::Overflow);
        if self.host_copies.is_some() || self.neural.is_none() {
            return Err(error(WorkingMemoryError::IdentityMismatch));
        }
        let transfers = self.host_transfers.ok_or_else(unknown)?;
        let count = transfers.copies;
        let bytes = transfers.bytes;
        let source = PreparedSourceCopies::prepare(layout, transfers, additional_controls, device)?;
        let PreparedSourceCopies { copies, rank, controls, .. } = source;
        let HostCopies { traversal, dispatch, direct_graph_extents,
            aggregate_traversal, aggregate_dispatch, .. } = copies;
        let completes_cpu_callbacks = self.retains_neural_bank_through_cpu_completion();
        for row in &mut self.records {
            let mut limits = row.traversal.ok_or_else(unknown)?.limits();
            let graph = row.graph.ok_or_else(unknown)?;
            let storage = row.mutable_storage.as_mut().ok_or_else(unknown)?;
            let equation_dispatch = row.dispatch.ok_or_else(unknown)?;
            if equation_dispatch.cpu_entries != 0 && !completes_cpu_callbacks {
                return Err(unknown());
            }
            // A later group can still visit submitted source descriptors. These
            // are traversal slots, not repeated Copy worker or buffer births.
            limits.arrays = limits
                .arrays
                .checked_add(count.checked_mul(2).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
            limits.input_edges = limits.input_edges.checked_add(count).ok_or_else(overflow)?;
            row.traversal =
                Some(OperationEvent::eval_traversal_layout(limits).ok_or_else(unknown)?);
            row.graph = Some(
                OperationEvent::resident_graph_layout_with_shells(
                    graph.primitives().checked_add(count).ok_or_else(overflow)?,
                    graph.seeds().checked_add(count).ok_or_else(overflow)?,
                    graph.maximum_rank().max(rank),
                    graph.maximum_operands().max(4),
                    // Actual native lease binding clones each selected Array
                    // once into its module. Neutral parameter construction has
                    // no backend C shell, so this is an independent producer.
                    graph
                        .additional_shells()
                        .checked_add(transfers.binding_shells)
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(unknown)?,
            );
            // Completed source descriptors extend the enclosing traversal;
            // their native copy workers remain separately priced below.
            let (equation_dispatch, equation_controls) = source.expand_equation_dispatch(
                equation_dispatch, limits, row.graph.ok_or_else(unknown)?.maximum_operands())?;
            row.dispatch = Some(equation_dispatch);
            // Original Metal page padding is applied once downstream to this
            // raw requested total and the actual independent birth population.
            storage.mutable_bytes = storage
                .mutable_bytes
                .checked_add(bytes)
                .ok_or_else(overflow)?;
            storage.maximum_births = storage
                .maximum_births
                .checked_add(count)
                .ok_or_else(overflow)?;
            row.maximum_rank = row.maximum_rank.max(rank);
            row.host_primitive_nodes = row
                .host_primitive_nodes
                .checked_add(count)
                .ok_or_else(overflow)?;
            row.query_controls = Some(
                row.query_controls
                    .ok_or_else(unknown)?
                    .checked_add(controls)
                    .and_then(|n| n.checked_add(equation_controls))
                    .ok_or_else(overflow)?,
            );
            // completion_for adds these independently priced source attempts.
            row.nested_completions
                .checked_add(count)
                .and_then(|n| n.checked_add(transfers.per_forward))
                .ok_or_else(overflow)?;
        }
        self.host_copies = Some(HostCopies {
            parameters: None,
            per_forward: count,
            traversal,
            dispatch,
            direct_graph_extents,
            aggregate_traversal,
            aggregate_dispatch,
        });
        Ok(())
    }
    pub(crate) fn has_host_copy_recipe(&self) -> bool {
        self.host_copies.is_some_and(|copies| copies.parameters.is_some())
    }
}

impl HostTransfers {
    fn from_windows(
        forwards: usize,
        transfers: usize,
        waits: usize,
        roots: usize,
        attempts_per_window: usize,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
    ) -> Option<Self> {
        if forwards == 0 || transfers % forwards != 0 || waits % forwards != 0 {
            return None;
        }
        let mut copies = 0usize;
        let mut binding_shells = 0usize;
        let mut bytes = 0u64;
        for window in windows.iter().filter(|w| w.requested != 0) {
            binding_shells =
                binding_shells.checked_add(window.bindings.checked_mul(attempts_per_window)?)?;
            // This is the actual finite canonical closure, including owners
            // outside the requested range. Warm/missing attempts are the same
            // slots used by ResidencyPopulation, not a new multiplier.
            copies =
                copies.checked_add(window.physical_bindings.checked_mul(attempts_per_window)?)?;
            bytes = bytes.checked_add(
                window
                    .physical_bytes
                    .checked_mul(u64::try_from(attempts_per_window).ok()?)?,
            )?;
        }
        roots.checked_add(1)?;
        Some(Self {
            per_forward: transfers / forwards,
            waits_per_forward: waits / forwards,
            roots,
            attempts_per_window,
            copies,
            binding_shells,
            bytes,
        })
    }
}
impl ResidentNativeRecipe {
    pub(crate) fn bind_host_transfer_population(
        &mut self,
        geometry: InferenceGeometry,
        forwards: usize,
        transfers: usize,
        waits: usize,
        roots: usize,
        attempts_per_window: usize,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
    ) -> Result<(), Error> {
        if geometry != self.plan.geometry()
            || Some(forwards) != self.plan.generation_forward_count()
            || self.host_transfers.is_some()
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        self.host_transfers = Some(
            HostTransfers::from_windows(
                forwards,
                transfers,
                waits,
                roots,
                attempts_per_window,
                windows,
            )
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        );
        Ok(())
    }
    pub(crate) fn matches_host_transfer_population(
        &self,
        forwards: usize,
        transfers: usize,
        waits: usize,
        roots: usize,
        attempts_per_window: usize,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
    ) -> bool {
        Some(forwards) == self.plan.generation_forward_count()
            && self.host_transfers.is_some()
            && self.host_transfers
                == HostTransfers::from_windows(
                    forwards,
                    transfers,
                    waits,
                    roots,
                    attempts_per_window,
                    windows,
                )
    }
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tests;
