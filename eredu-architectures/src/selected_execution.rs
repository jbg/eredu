//! Total backend-neutral selections retained between cold admission and materialization.

use std::collections::BTreeSet;

use eredu_core::{
    artifact::ArtifactAdmissionToken, CollectiveGroupId, PreparationAdmission, SessionCapabilities,
};
use eredu_runtime::{
    CommunicationManifest, ParameterBankResidency, PipelineActivationDtype,
    ReplicatedTextRequirements, SelectedReplicatedTextRealization, SelectedSpeculativeRealization,
};

use crate::{
    configuration::PredictionExtensionPlan,
    partitioned_execution::SelectedPartitionedAdmission,
    replicated_text::{
        CompositeTextRequirements, SelectedCompositeTextRealization,
        SelectedReplicatedTextExecution, SelectedReplicatedTextExecutionDispatcher,
    },
    RoutedTextRequirements, SelectedRoutedTextRealization,
};

/// Selected direct partitioned execution, including admitted communication and ownership.
pub type SelectedDensePartitionedExecution =
    SelectedPartitionedAdmission<SelectedReplicatedTextRealization, ReplicatedTextRequirements>;

/// Selected routed partitioned execution, including admitted communication and ownership.
pub type SelectedRoutedPartitionedExecution =
    SelectedPartitionedAdmission<SelectedRoutedTextRealization, RoutedTextRequirements>;

/// Selected composite partitioned execution, including admitted communication and ownership.
pub type SelectedCompositePartitionedExecution =
    SelectedPartitionedAdmission<SelectedCompositeTextRealization, CompositeTextRequirements>;

type SelectedOrdinaryExecution = SelectedReplicatedTextExecution<
    SelectedReplicatedTextRealization,
    SelectedRoutedTextRealization,
    SelectedCompositeTextRealization,
>;

#[derive(Debug, Clone)]
enum SelectedExecutionKind {
    Replicated(SelectedReplicatedTextRealization),
    Routed(SelectedRoutedTextRealization),
    Composite(SelectedCompositeTextRealization),
    PartitionedDense(SelectedDensePartitionedExecution),
    PartitionedRouted(SelectedRoutedPartitionedExecution),
    PartitionedComposite(SelectedCompositePartitionedExecution),
}

/// An owned adapter for exactly one selected execution branch.
///
/// A backend implements this trait on a concrete materializer. Dispatch is
/// monomorphized and consumes both the authoritative selection and the
/// materializer, so no branch reconstruction or dynamic dispatch is needed.
pub trait SelectedExecutionDispatcher: Sized {
    /// Completed materialization output.
    type Output;
    /// Materialization failure.
    type Error;

    /// Materializes ordinary replicated text execution.
    fn replicated(
        self,
        selected: SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes ordinary routed text execution.
    fn routed(self, selected: SelectedRoutedTextRealization) -> Result<Self::Output, Self::Error>;

    /// Materializes ordinary composite text execution.
    fn composite(
        self,
        selected: SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes direct partitioned text execution.
    fn partitioned_dense(
        self,
        selected: SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes routed partitioned text execution.
    fn partitioned_routed(
        self,
        selected: SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;

    /// Materializes composite partitioned text execution.
    fn partitioned_composite(
        self,
        selected: SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, Self::Error>;
}

/// One authoritative backend-neutral execution selection.
///
/// The semantic branch is private. Backends can only consume it through the
/// typed dispatcher, preventing a materializer from substituting or rebuilding
/// the selection after admission.
#[derive(Debug, Clone)]
pub struct SelectedExecution {
    kind: Box<SelectedExecutionKind>,
}

impl SelectedExecution {
    /// Exact portable rank topology of a selected partition, when present.
    pub fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().topology())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().topology())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().topology())
            }
            _ => None,
        }
    }

    /// Exact selected processor policy for composite execution.
    pub fn processor(&self) -> Option<&eredu_runtime::SelectedProcessorExecution> {
        match self.kind.as_ref() {
            SelectedExecutionKind::Composite(selected) => Some(selected.processor()),
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.base().processor())
            }
            _ => None,
        }
    }

    pub(crate) fn replicated(selected: SelectedReplicatedTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Replicated(selected)),
        }
    }

    pub(crate) fn routed(selected: SelectedRoutedTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Routed(selected)),
        }
    }

    pub(crate) fn composite(selected: SelectedCompositeTextRealization) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::Composite(selected)),
        }
    }

    pub(crate) fn partitioned_dense(selected: SelectedDensePartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedDense(selected)),
        }
    }

    pub(crate) fn partitioned_routed(selected: SelectedRoutedPartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedRouted(selected)),
        }
    }

    pub(crate) fn partitioned_composite(selected: SelectedCompositePartitionedExecution) -> Self {
        Self {
            kind: Box::new(SelectedExecutionKind::PartitionedComposite(selected)),
        }
    }

    pub(crate) fn ordinary(selected: SelectedOrdinaryExecution) -> Self {
        struct IntoTotal;

        impl
            SelectedReplicatedTextExecutionDispatcher<
                SelectedReplicatedTextRealization,
                SelectedRoutedTextRealization,
                SelectedCompositeTextRealization,
            > for IntoTotal
        {
            type Output = SelectedExecution;
            type Error = std::convert::Infallible;

            fn replicated(
                self,
                selected: SelectedReplicatedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::replicated(selected))
            }

            fn routed(
                self,
                selected: SelectedRoutedTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::routed(selected))
            }

            fn composite(
                self,
                selected: SelectedCompositeTextRealization,
            ) -> Result<Self::Output, Self::Error> {
                Ok(SelectedExecution::composite(selected))
            }
        }

        selected
            .dispatch(IntoTotal)
            .expect("ordinary execution conversion is infallible")
    }

    /// Invokes exactly one backend materializer with the owned selected branch.
    pub fn dispatch<D>(self, dispatcher: D) -> Result<D::Output, D::Error>
    where
        D: SelectedExecutionDispatcher,
    {
        match *self.kind {
            SelectedExecutionKind::Replicated(selected) => dispatcher.replicated(selected),
            SelectedExecutionKind::Routed(selected) => dispatcher.routed(selected),
            SelectedExecutionKind::Composite(selected) => dispatcher.composite(selected),
            SelectedExecutionKind::PartitionedDense(selected) => {
                dispatcher.partitioned_dense(selected)
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                dispatcher.partitioned_routed(selected)
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                dispatcher.partitioned_composite(selected)
            }
        }
    }

    /// Returns the shared text-session realization selected for every execution class.
    pub fn text_realization(&self) -> &SelectedReplicatedTextRealization {
        match self.kind.as_ref() {
            SelectedExecutionKind::Replicated(selected) => selected,
            SelectedExecutionKind::Routed(selected) => selected.text(),
            SelectedExecutionKind::Composite(selected) => selected.execution(),
            SelectedExecutionKind::PartitionedDense(selected) => selected.base(),
            SelectedExecutionKind::PartitionedRouted(selected) => selected.base().text(),
            SelectedExecutionKind::PartitionedComposite(selected) => selected.base().execution(),
        }
    }

    /// Returns parameter targets moved into independently resident routed banks.
    ///
    /// Partitioned materialization owns exact rank-local tasks, so it does not
    /// use the ordinary whole-model exclusion projection.
    pub fn bounded_residency_exclusions(&self) -> BTreeSet<String> {
        let SelectedExecutionKind::Routed(selected) = self.kind.as_ref() else {
            return BTreeSet::new();
        };
        if !matches!(
            selected.bank_residency(),
            ParameterBankResidency::IndependentCache(_)
        ) {
            return BTreeSet::new();
        }
        selected
            .addressable_members()
            .iter()
            .flat_map(|member| member.parameters())
            .map(|parameter| parameter.task().name().to_owned())
            .collect()
    }

    /// Returns the selected text realization and ordinary materialization exclusions.
    pub fn selected_bounded_residency(
        &self,
    ) -> (SelectedReplicatedTextRealization, BTreeSet<String>) {
        (
            self.text_realization().clone(),
            self.bounded_residency_exclusions(),
        )
    }

    /// Returns the admitted communication manifest for partitioned execution.
    pub fn communication_manifest(&self) -> Option<&CommunicationManifest> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().communication())
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Returns the admitted pipeline activation dtype for partitioned execution.
    pub fn partitioned_activation_dtype(&self) -> Option<PipelineActivationDtype> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                Some(selected.requirements().activation_dtype())
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Returns the selected session-wide publication group for partitioned execution.
    pub fn partitioned_session_group(&self) -> Option<CollectiveGroupId> {
        match self.kind.as_ref() {
            SelectedExecutionKind::PartitionedDense(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::PartitionedRouted(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::PartitionedComposite(selected) => {
                selected.requirements().session_group()
            }
            SelectedExecutionKind::Replicated(_)
            | SelectedExecutionKind::Routed(_)
            | SelectedExecutionKind::Composite(_) => None,
        }
    }

    /// Reports whether the selected execution consumes a composite media projector.
    pub fn allows_media_projector(&self) -> bool {
        matches!(
            self.kind.as_ref(),
            SelectedExecutionKind::Composite(_) | SelectedExecutionKind::PartitionedComposite(_)
        )
    }
}

/// Complete neutral preparation selection retained until backend materialization.
///
/// This value deliberately contains no device, stream, rank binding, native
/// completion object, or backend-private token.
#[derive(Debug, Clone)]
pub struct SelectedPreparation {
    admission_token: ArtifactAdmissionToken,
    execution: SelectedExecution,
    admission: PreparationAdmission,
    prediction_extension: Option<PredictionExtensionPlan>,
    prediction_realization: Option<SelectedSpeculativeRealization>,
}

impl SelectedPreparation {
    pub(crate) const fn new(
        admission_token: ArtifactAdmissionToken,
        execution: SelectedExecution,
        admission: PreparationAdmission,
        prediction_extension: Option<PredictionExtensionPlan>,
        prediction_realization: Option<SelectedSpeculativeRealization>,
    ) -> Self {
        Self {
            admission_token,
            execution,
            admission,
            prediction_extension,
            prediction_realization,
        }
    }

    /// Opaque inspection origin retained by total cold selection.
    pub fn admission_token(&self) -> ArtifactAdmissionToken {
        self.admission_token.clone()
    }

    /// Returns the total execution selected before native work.
    pub const fn execution(&self) -> &SelectedExecution {
        &self.execution
    }

    /// Returns the retained portable admission proof.
    pub const fn admission(&self) -> PreparationAdmission {
        self.admission
    }

    /// Returns session capabilities admitted for the exact selected preparation.
    pub const fn session_capabilities(&self) -> SessionCapabilities {
        self.admission.session_capabilities()
    }

    /// Returns the selected embedded prediction-extension architecture, when present.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.prediction_extension.as_ref()
    }

    /// Returns the selected speculative realization for the embedded extension, when present.
    pub const fn prediction_realization(&self) -> Option<&SelectedSpeculativeRealization> {
        self.prediction_realization.as_ref()
    }

    /// Returns the shared text-session realization selected for every execution class.
    pub fn text_realization(&self) -> &SelectedReplicatedTextRealization {
        self.execution.text_realization()
    }

    /// Returns the selected text realization and ordinary materialization exclusions.
    pub fn selected_bounded_residency(
        &self,
    ) -> (SelectedReplicatedTextRealization, BTreeSet<String>) {
        self.execution.selected_bounded_residency()
    }

    /// Returns the admitted communication manifest for partitioned execution.
    pub fn communication_manifest(&self) -> Option<&CommunicationManifest> {
        self.execution.communication_manifest()
    }

    /// Returns the admitted pipeline activation dtype for partitioned execution.
    pub fn partitioned_activation_dtype(&self) -> Option<PipelineActivationDtype> {
        self.execution.partitioned_activation_dtype()
    }

    /// Returns the selected session-wide publication group for partitioned execution.
    pub fn partitioned_session_group(&self) -> Option<CollectiveGroupId> {
        self.execution.partitioned_session_group()
    }

    /// Reports whether the selected execution consumes a composite media projector.
    pub fn allows_media_projector(&self) -> bool {
        self.execution.allows_media_projector()
    }

    /// Consumes the selection into neutral materialization inputs.
    pub(crate) fn into_parts(
        self,
    ) -> (
        SelectedExecution,
        PreparationAdmission,
        Option<PredictionExtensionPlan>,
        Option<SelectedSpeculativeRealization>,
    ) {
        (
            self.execution,
            self.admission,
            self.prediction_extension,
            self.prediction_realization,
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn public_selection_types_are_owned_and_thread_safe() {
        fn assert_owned<T: Clone + Send + Sync + 'static>() {}

        assert_owned::<super::SelectedExecution>();
        assert_owned::<super::SelectedPreparation>();
    }
}
