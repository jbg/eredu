//! Fixed original numerical programs. Public sampler callbacks remain separate.
use super::*;
use crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources;
use crate::composition::mlx::model::retain_planning_error;
use crate::composition::mlx::speculative::autoregressive::{
    CompletedNumericalPrefill, StateStream,
    input_readout::CompletedNumericalReadout,
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::speculative::numerical as program;
use eredu_runtime::working_memory::{
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeNumericalBudgetCustody,
    SpeculativeNumericalSource, OriginalSpeculativeRegisteredSource,
};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};
mod native;
mod model;
mod readout;
mod tensor;
mod registered;
mod transfer;
pub(crate) use transfer::{copy_key_to,copy_value_to};
pub(crate) use registered::token_input as registered_copy_input;
pub(crate) use tensor::{token_ids, tensor_concatenate, tensor_range, tensor_axis_range, tensor_axis_range_at, registered_range, RegisteredTensorSource, CompletedTensorSource};
pub(crate) use readout::{completed_logits, logits_row, completed_logits_at, logits_row_at};
pub(crate) use model::PendingModelLogits;
mod policy;
mod capture;
pub(crate) use capture::CaptureSource;
mod choice;
mod adaptive;
pub(crate) use adaptive::commit_probability;
mod random;
mod snapshot;
pub(super) use snapshot::SnapshotContext;
pub(crate) use random::{create_key, next_key, key_at, sample_stochastic, sample_stochastic_at, sample_unit_interval, OriginalNumericalKey};
pub(crate) use choice::{sample_greedy,sample_greedy_at, commit_without_mutation, validate_commit_value};
pub(crate) use policy::{process_policy,process_policy_at,process_policy_with_capture,process_policy_with_capture_at,validate_controller_value};
pub(super) mod operators;
pub(crate) use native::{NumericalOutput, NumericalProducer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Meaning {
    RandomKey,
    TokenIds,
    Capture,
    Logits,
    Probabilities,
    Difference,
}
#[derive(Clone)]
enum Provenance {
    Model(OriginalSpeculativeBudgetCustody),
    Numerical(OriginalSpeculativeNumericalBudgetCustody),
    Registered(OriginalSpeculativeRegisteredSource),
}
impl Provenance {
    fn source(&self) -> SpeculativeNumericalSource<'_> {
        match self {
            Self::Model(value) => SpeculativeNumericalSource::Model(value),
            Self::Numerical(value) => SpeculativeNumericalSource::Numerical(value),
            Self::Registered(value) => SpeculativeNumericalSource::Registered(value),
        }
    }
}
#[derive(Clone)]
enum ValueStream {
    Model(StateStream),
    Embedded(safemlx::PreparedStreamCopy<model::StreamOwner>),
    Numerical(safemlx::PreparedStreamCopy<OriginalSpeculativeNumericalBudgetCustody>),
}
impl std::ops::Deref for ValueStream {
    type Target = Stream;
    fn deref(&self) -> &Stream {
        match self { Self::Model(stream) => stream, Self::Numerical(stream) => stream.as_stream(), Self::Embedded(stream) => stream.as_stream() }
    }
}
// Fixed source retention in the existing numerical owner. Both operands retire
// after the result Array, including failed/escaped concatenation destinations.
enum RetainedValues {
    One(OriginalNumericalValue),
    Two([OriginalNumericalValue; 2]),
    // Imported tensor readouts keep the actual copy/view proof independently
    // of the account that owns their shared physical backing.
    Evidence(eredu_architectures::speculative_execution::PreparedEmbeddedEvidence),
}
struct Value {
    array: Array,
    // Only the successful native phase installs this exact originating budget.
    // A published independent copy instead retains its existing registry custody.
    original_budget: Option<safemlx::OriginalBufferBudget>,
    stream: ValueStream,
    meaning: Meaning,
    provenance: Provenance,
    funding: WorkspaceMetadataFunding,
    // A copied Array retires before its independent registered-copy Q/H.
    _copy: Option<crate::backend::array_copy::RegisteredArrayCopyCustody>,
    _snapshot_host: Option<eredu_core::HostPreparationAuthority>,
    // A static row is a view into its actual completed source. The numerical
    // phase owns its graph/record custody and this alias owns the source Q/H.
    _readout_source: Option<RetainedValues>,
}
/// Closed completed value. There is no raw-array/role constructor or extraction;
/// its two producers are an actual once-only readout and this numerical worker.
pub(crate) struct OriginalNumericalValue(Option<Rc<Value>>, Option<eredu_runtime::generation::PreparedControllerChoice>);
impl Clone for OriginalNumericalValue {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(
            self.0.as_ref().expect("live numerical value"),
        )), self.1.clone())
    }
}
impl Drop for OriginalNumericalValue {
    fn drop(&mut self) {
        drop(self.1.take());
        if let Some(value) = self.0.take() {
            drop(Rc::into_inner(value));
        }
    }
}
impl std::fmt::Debug for OriginalNumericalValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalNumericalValue")
    }
}
impl OriginalNumericalValue {
    pub(in crate::composition::mlx::speculative) fn from_prefill(
        source: CompletedNumericalPrefill,
    ) -> Result<Self, Error> {
        let bytes = value_control_bytes().ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
        source
            .funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let (array, stream, custody, funding) = source.into_parts();
        Ok(Self(Some(Rc::new(Value {
            array,
            original_budget: None,
            stream: ValueStream::Model(stream),
            meaning: Meaning::Logits,
            provenance: Provenance::Model(custody),
            funding,
            _copy: None,
            _snapshot_host: None,
            _readout_source: None,
        })), None))
    }
    pub(in crate::composition::mlx::speculative) fn from_readout(
        readout: CompletedNumericalReadout,
    ) -> Result<Self, Error> {
        let bytes = value_control_bytes().ok_or(Error::WorkspacePlanning(
            WorkspaceMetadataFundingError::Overflow,
        ))?;
        readout
            .funding()
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        // Every fallible preparation precedes the move. The closed Rc frees
        // its shell before the value, C handle and funding/account retire.
        let (array, stream, custody, funding) = readout.into_parts();
        Ok(Self(Some(Rc::new(Value {
            array,
            original_budget: None,
            stream: ValueStream::Model(stream),
            meaning: Meaning::Logits,
            provenance: Provenance::Model(custody),
            funding,
            _copy: None,
            _snapshot_host: None,
            _readout_source: None,
        })), None))
    }
    /// Borrow only the account retained by this completed value after checking
    /// its exact request provenance. This does not authorize native work.
    pub(super) fn validate_consumer(
        &self,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<&WorkspaceMetadataFunding, Error> {
        let value = self.value();
        if !value.provenance.source().belongs_to_request(sources.request()) {
            // No diagnostic allocation or foreign funding substitution. The
            // borrowed value retains its own payload through this fixed refusal.
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        Ok(&value.funding)
    }
    fn with_controller_choice(mut self,choice:Option<eredu_runtime::generation::PreparedControllerChoice>)->Self {
        self.1=choice;
        self
    }
    pub(in crate::composition::mlx::speculative::sampling) fn controller_choice(&self)->Option<&eredu_runtime::generation::PreparedControllerChoice> { self.1.as_ref() }
    fn with_original_budget(mut self, budget: safemlx::OriginalBufferBudget) -> Self {
        Rc::get_mut(self.0.as_mut().expect("live numerical value"))
            .expect("unpublished completed key").original_budget = Some(budget);
        self
    }

    fn value(&self) -> &Value {
        self.0.as_deref().expect("live numerical value")
    }
    fn completed(
        array: Array,
        stream: ValueStream,
        meaning: Meaning,
        custody: OriginalSpeculativeNumericalBudgetCustody,
        funding: WorkspaceMetadataFunding,
    ) -> Self {
        Self(Some(Rc::new(Value {
            array,
            original_budget: None,
            stream,
            meaning,
            provenance: Provenance::Numerical(custody),
            funding,
            _copy: None,
            _snapshot_host: None,
            _readout_source: None,
        })), None)
    }
}
fn rc_bytes<T>() -> Option<usize> {
    Layout::new::<[Cell<usize>; 2]>()
        .extend(Layout::new::<T>())
        .ok()
        .map(|layout| layout.0.pad_to_align().size())
}
fn value_control_bytes() -> Option<usize> {
    let parts = [
        eredu_runtime::generation::PreparedControllerChoice::metadata_bytes(),
        size_of::<Value>(),
        size_of::<OriginalNumericalValue>(),
        size_of::<CompletedNumericalReadout>(),
        size_of::<CompletedNumericalPrefill>(),
        size_of::<Result<OriginalNumericalValue, Error>>(),
        rc_bytes::<Value>()?,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
pub(in crate::composition::mlx::speculative) mod native_tests;

pub(in crate::composition::mlx::speculative) use tensor::native_axis_range;
