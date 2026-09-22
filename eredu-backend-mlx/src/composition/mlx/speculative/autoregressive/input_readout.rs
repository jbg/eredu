//! Admitted input and output views for the existing independent decoder call.
use super::workspace::ActiveSpeculativeInvocation;
use super::*;
use crate::backend::nn::workspace::{AutoregressiveReadoutRecipe, ResidentExecutionMechanisms};
use crate::composition::mlx::replicated_text::{
    AutoregressiveSequenceCompletion, AutoregressiveStateRoots,
};
use eredu_nn::{
    Index, Tensor,
    workspace::{WorkspaceContext, WorkspaceDtype, WorkspaceMetadataError, WorkspaceTensor},
};
use eredu_runtime::{
    speculative::autoregressive::AutoregressiveInvocation, working_memory::OriginalSpeculativeRole,
};
use safemlx::{OriginalPromptInputFacts, OriginalScopeObserver, PreparedInputRuntime};
use std::{
    alloc::Layout,
    cell::RefCell,
    mem::{size_of, size_of_val},
};

/// Descriptive source-derived populations. Native activation still consumes an
/// accepted occurrence and its same-request role in the enclosing compiler.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AutoregressiveIoPlan {
    invocation: AutoregressiveInvocation,
    vocabulary: usize,
    rows: usize,
    input: OriginalPromptInputFacts,
    readout: Option<AutoregressiveReadoutRecipe>,
    controls: usize,
}
impl AutoregressiveIoPlan {
    pub(crate) fn inspect(
        runtime: &PreparedInputRuntime,
        invocation: AutoregressiveInvocation,
        vocabulary: usize,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "speculative input/readout geometry differs from its invocation"
            ))
        };
        if invocation.execution_pass() != eredu_runtime::ExpertPass::Decode || vocabulary == 0 {
            return Err(invalid());
        }
        let positions = i32::try_from(invocation.positions()).map_err(|_| invalid())?;
        let width = i32::try_from(vocabulary).map_err(|_| invalid())?;
        let rows = match invocation.pass() {
            AutoregressivePass::Proposal if positions == 1 => 1,
            AutoregressivePass::Verification => invocation.positions(),
            AutoregressivePass::TargetCommit | AutoregressivePass::DraftCommit => 0,
            _ => return Err(invalid()),
        };
        let input = OriginalPromptInputFacts::inspect(runtime, invocation.positions())
            .map_err(|cause| context.metadata_source(cause))?;
        let readout = if rows == 0 {
            None
        } else {
            // A separate recording identity prevents this suffix from changing
            // the already-finished model trace. Its actual constructors debit
            // the same cumulative diagnostic account before allocation.
            let funding = context
                .metadata_funding()
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let readout_context = mechanism.context(funding)?;
            let source = WorkspaceTensor::existing(
                readout_context.layout(&[1, positions, width], WorkspaceDtype::Float32)?,
                &readout_context,
            )?;
            readout_context.begin_span();
            let mut outputs = readout_context.metadata_vec(rows)?;
            for position in 0..rows {
                // This is the same static index as the ordinary native call:
                // one rank-preserving Slice followed by removal of axis one.
                outputs.push(source.index(
                    &[Index::Full, Index::At(position as i32), Index::Full],
                    &readout_context,
                )?);
            }
            let report = readout_context.finish_report(&outputs)?;
            Some(AutoregressiveReadoutRecipe::inspect(
                &report,
                rows,
                invocation.positions(),
                mechanism,
                &readout_context,
            )?)
        };
        let controls = runtime_controls(rows, input).ok_or(WorkspaceMetadataError::Overflow)?;
        Ok(Self {
            invocation,
            vocabulary,
            rows,
            input,
            readout,
            controls,
        })
    }
    pub(crate) fn input_facts(&self) -> Option<OriginalPromptInputFacts> {
        Some(self.input)
    }
    pub(crate) fn readout_recipe(&self) -> Option<&AutoregressiveReadoutRecipe> {
        self.readout.as_ref()
    }
    pub(crate) fn control_bytes(&self) -> usize {
        self.controls
    }
    pub(crate) fn root_count(&self) -> usize {
        self.rows + 1
    }
    pub(crate) fn prepare(
        self,
        role: OriginalSpeculativeRole,
    ) -> Result<PreparedAutoregressiveIo, IoPreparationError> {
        if role.invocation() != self.invocation {
            return Err(IoPreparationError {
                cause: IoPreparationCause::Source,
                role,
            });
        }
        let mut rows = Vec::new();
        if let Err(cause) = rows.try_reserve_exact(self.rows) {
            return Err(IoPreparationError {
                cause: IoPreparationCause::Reserve(cause),
                role,
            });
        }
        rows.resize_with(self.rows, || None);
        Ok(PreparedAutoregressiveIo {
            plan: self,
            rows,
            input_spent: false,
            completed: false,
            role,
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum IoPreparationCause {
    #[error("speculative input/readout role differs from its plan")]
    Source,
    #[error("speculative readout slot allocation failed")]
    Reserve(#[source] std::collections::TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct IoPreparationError {
    #[source]
    cause: IoPreparationCause,
    // The original accepted account outlives the exact error and its prefix.
    role: OriginalSpeculativeRole,
}

pub(crate) struct PreparedAutoregressiveIo {
    plan: AutoregressiveIoPlan,
    rows: Vec<Option<Array>>,
    input_spent: bool,
    completed: bool,
    role: OriginalSpeculativeRole,
}
impl PreparedAutoregressiveIo {
    pub(super) fn input(
        &mut self,
        tokens: &[u32],
        pass: AutoregressivePass,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer) {
            return Err(observer.domain_error());
        }
        if self.input_spent
            || pass != self.plan.invocation.pass()
            || tokens.len() != self.plan.input.elements()
        {
            return Err(observer.invalid_input_error());
        }
        // The occurrence is spent before the eager constructor can attempt any
        // allocation. A failed native input never receives another attempt.
        self.input_spent = true;
        Array::try_from_original_prompt_ids(tokens)
    }
    pub(super) fn sequence_completion<'a>(
        &'a mut self,
        invocation: &'a ActiveSpeculativeInvocation,
    ) -> SequenceCompletion<'a> {
        SequenceCompletion {
            io: self,
            invocation,
        }
    }
    fn complete_transaction(
        &mut self,
        logits: &Array,
        stream: &Stream,
        invocation: &ActiveSpeculativeInvocation,
        state: &dyn AutoregressiveStateRoots,
    ) -> Result<(), Error> {
        let observer = invocation.observer();
        if self.completed || !self.role.same_role(invocation.role()) {
            return Err(observer.domain_error().into());
        }
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer) {
            return Err(observer.domain_error().into());
        }
        {
            let descriptor = logits.try_descriptor().map_err(|cause| match cause {
                safemlx::ArrayDescriptorError::RuntimeBusy => observer
                    .observation_error(safemlx::ScopedSubmissionProgress::Busy)
                    .expect("Busy is refusal"),
                safemlx::ArrayDescriptorError::SourceChanged => observer.domain_error(),
                _ => observer.invalid_input_error(),
            })?;
            if !self.input_spent
                || descriptor.shape()
                    != [
                        1,
                        self.plan.invocation.positions() as i32,
                        self.plan.vocabulary as i32,
                    ]
                || !matches!(
                    descriptor.facts().dtype(),
                    safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float32
                )
            {
                return Err(observer.invalid_input_error().into());
            }
        }
        if !self.rows.is_empty() {
            invocation.begin_readout_construction()?;
        }
        for (position, slot) in self.rows.iter_mut().enumerate() {
            *slot = Some(logits.try_index_device((.., position as i32, ..), stream)?);
        }
        let completion = CompleteOutput {
            invocation,
            state,
            stream,
        };
        completion.complete(logits, &self.rows)?;
        self.completed = true;
        Ok(())
    }
    pub(super) fn finish(
        self,
        logits: Array,
        stream: &StateStream,
        invocation: &ActiveSpeculativeInvocation,
    ) -> Result<MlxAutoregressiveOutput, Error> {
        if !self.completed || !self.role.same_role(invocation.role()) {
            return Err(invocation.observer().domain_error().into());
        }
        Ok(MlxAutoregressiveOutput {
            logits,
            stream: stream.clone(),
            original: Some(CompletedReadouts {
                rows: RefCell::new(self.rows),
                observer: invocation.observer().clone(),
                role: self.role,
                funding: invocation.metadata_funding(),
            }),
        })
    }
}
pub(super) struct SequenceCompletion<'a> {
    io: &'a mut PreparedAutoregressiveIo,
    invocation: &'a ActiveSpeculativeInvocation,
}
impl AutoregressiveSequenceCompletion for SequenceCompletion<'_> {
    fn active_invocation(&self) -> ActiveSpeculativeInvocation {
        self.invocation.clone()
    }
    fn metadata_context(&self) -> WorkspaceContext {
        self.invocation.metadata_context()
    }

    fn role(&self) -> &OriginalSpeculativeRole {
        self.invocation.role()
    }
    fn metadata_funding(&self) -> eredu_nn::workspace::HostMetadataFunding {
        self.invocation.metadata_funding()
    }
    fn take_checkpoint(&mut self) -> Result<MlxPredictionTargetState, Error> {
        self.invocation.take_checkpoint()
    }
    fn complete(
        &mut self,
        output: Option<&Array>,
        state: &dyn AutoregressiveStateRoots,
        stream: &Stream,
    ) -> Result<(), Error> {
        let output = output.ok_or(Error::InvalidOperation("decode completion requires scores"))?;
        self.io
            .complete_transaction(output, stream, self.invocation, state)
    }
}

struct CompleteOutput<'a> {
    invocation: &'a ActiveSpeculativeInvocation,
    state: &'a dyn AutoregressiveStateRoots,
    stream: &'a Stream,
}
impl CompleteOutput<'_> {
    fn complete(&self, logits: &Array, rows: &[Option<Array>]) -> Result<(), Error> {
        let select: fn(&Option<Array>) -> Option<&Array> = Option::as_ref;
        self.invocation.complete_sequence_roots(
            std::iter::once(logits).chain(rows.iter().filter_map(select)),
            self.state,
            self.stream,
        )
    }
}

/// Every actual row retires before the shared original account. No Rc/Weak or
/// mutable native scope escapes; the private constructor requires completion.
pub(super) struct CompletedReadouts {
    rows: RefCell<Vec<Option<Array>>>,
    observer: OriginalScopeObserver,
    role: OriginalSpeculativeRole,
    funding: eredu_nn::workspace::HostMetadataFunding,
}
/// Only the real once-only completed row worker can create this source proof.
pub(crate) struct CompletedNumericalReadout {
    value: Array,
    stream: StateStream,
    custody: eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody,
    funding: eredu_nn::workspace::HostMetadataFunding,
}
impl CompletedNumericalReadout {
    pub(in crate::composition::mlx::speculative) fn funding(
        &self,
    ) -> &eredu_nn::workspace::HostMetadataFunding {
        &self.funding
    }
    pub(in crate::composition::mlx::speculative) fn into_parts(
        self,
    ) -> (
        Array,
        StateStream,
        eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody,
        eredu_nn::workspace::HostMetadataFunding,
    ) {
        (self.value, self.stream, self.custody, self.funding)
    }
}
impl CompletedReadouts {
    pub(super) fn take_numerical(
        &self,
        position: usize,
        stream: &StateStream,
    ) -> Result<super::super::sampling::numerical::OriginalNumericalValue, Error> {
        if !matches!(stream, StateStream::Original(_)) {
            return Err(self.observer.domain_error().into());
        }
        let value = self.take(position).map_err(Error::from)?;
        super::super::sampling::numerical::OriginalNumericalValue::from_readout(
            CompletedNumericalReadout {
                value,
                stream: stream.clone(),
                custody: self.role.budget_custody(),
                funding: self.funding.clone(),
            },
        )
    }
    pub(super) fn role(&self) -> &OriginalSpeculativeRole {
        &self.role
    }
    pub(super) fn take(&self, position: usize) -> Result<Array, Exception> {
        let value = {
            let Ok(mut rows) = self.rows.try_borrow_mut() else {
                return Err(self
                    .observer
                    .observation_error(safemlx::ScopedSubmissionProgress::Busy)
                    .expect("Busy is refusal"));
            };
            rows.get_mut(position).and_then(Option::take)
        };
        value.ok_or_else(|| self.observer.invalid_input_error())
    }
}
fn runtime_controls(rows: usize, input: OriginalPromptInputFacts) -> Option<usize> {
    type ReadoutIter<'a> = std::iter::FilterMap<
        std::slice::Iter<'a, Option<Array>>,
        fn(&'a Option<Array>) -> Option<&'a Array>,
    >;
    let parts = [
        Layout::array::<Option<Array>>(rows).ok()?.size(),
        size_of::<AutoregressiveIoPlan>(),
        size_of::<PreparedAutoregressiveIo>(),
        size_of::<SequenceCompletion<'_>>(),
        size_of::<&mut dyn AutoregressiveSequenceCompletion>(),
        size_of::<&dyn AutoregressiveStateRoots>(),
        size_of::<Result<(), Error>>(),
        size_of::<CompletedReadouts>(),
        size_of::<MlxAutoregressiveOutput>(),
        size_of::<IoPreparationCause>(),
        size_of::<IoPreparationError>(),
        size_of::<Result<PreparedAutoregressiveIo, IoPreparationError>>(),
        size_of::<Result<MlxAutoregressiveOutput, Error>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Option<CompletedReadouts>>(),
        size_of::<CompleteOutput<'_>>(),
        size_of::<std::cell::RefMut<'_, Vec<Option<Array>>>>(),
        size_of::<std::slice::IterMut<'_, Option<Array>>>(),
        size_of::<std::iter::Enumerate<std::slice::IterMut<'_, Option<Array>>>>(),
        size_of::<std::iter::Chain<std::iter::Once<&Array>, ReadoutIter<'_>>>(),
        size_of::<(
            &mut Executable,
            &[u32],
            &mut MlxAutoregressiveState,
            AutoregressivePass,
            SpeculativeExecutionStreams<'_>,
        )>(),
        size_of::<(AutoregressiveIoPlan, OriginalSpeculativeRole)>(),
        OriginalScopeObserver::control_bytes()?,
        Array::descriptor_control_bytes()?,
        safemlx::ops::indexing::inline_basic_index_control_bytes()?.checked_mul(rows)?,
        input.control_bytes()?,
        // Slot, bank, model and final completion return the same typed error;
        // none creates an intermediate Exception source Box.
        size_of::<Error>().checked_mul(4)?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
