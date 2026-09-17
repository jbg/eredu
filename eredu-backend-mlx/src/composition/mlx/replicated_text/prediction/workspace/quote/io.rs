//! Actual eager signed token source and immediate sequential logits-row tail.
use super::*;
use crate::composition::mlx::model::retain_planning_error;
use eredu_architectures::{
    prediction_extension::equation::PredictionEquationOutput,
    prepared_execution::WorkspacePredictionEquationTails,
};
use eredu_nn::{Index, Tensor, workspace::WorkspaceDtype};
use eredu_runtime::working_memory::{InferenceSpanWorkspacePlan, OriginalEmbeddedSpeculativeRole};
use safemlx::{
    Array, OriginalPromptInputFacts, OriginalScopeObserver, PreparedArrayClone,
    PreparedInputRuntime,
};

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(super) enum IoError {
    #[error("prediction token/readout differs from its exact equation source")]
    Source,
    #[error("prediction scalar/readout source was consumed more than once")]
    Consumed,
}
/// Descriptive facts from the actual checked source. Preparation consumes this
/// once; neither geometry nor the fact object authorizes a native constructor.
pub(in crate::composition::mlx::replicated_text) struct PredictionIoPlan {
    workspace: EmbeddedInvocationWorkspace,
    scalar: Option<(u32, i32, OriginalPromptInputFacts)>,
    scalar_workspace: Option<WorkspaceTensor>,
    sequential: bool,
    token_quoted: bool,
    output_quoted: bool,
    completed_state_roots: Option<usize>,
    completion_output_roots: usize,
}
impl PredictionIoPlan {
    pub(super) fn inspect(
        equation: &PredictionEquation<&MlxTensor>,
        workspace: EmbeddedInvocationWorkspace,
        runtime: &PreparedInputRuntime,
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<Option<(u32, i32, OriginalPromptInputFacts)>>(),
            size_of::<Result<OriginalPromptInputFacts, safemlx::OriginalPromptInputCause>>(),
            size_of::<Result<i32, std::num::TryFromIntError>>(),
            size_of::<(
                &PredictionEquation<&MlxTensor>,
                EmbeddedInvocationWorkspace,
                &PreparedInputRuntime,
                &WorkspaceContext,
            )>(),
        ];
        context.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let token = match equation {
            PredictionEquation::Sequential { token, .. } => Some(*token),
            PredictionEquation::Fused { anchor, .. } => Some(*anchor),
            _ => None,
        };
        let scalar = if let Some(id) = token {
            let signed = i32::try_from(id).map_err(|_| context.metadata_source(IoError::Source))?;
            let facts = OriginalPromptInputFacts::inspect(runtime, 1)
                .map_err(|cause| context.metadata_source(cause))?;
            Some((id, signed, facts))
        } else {
            None
        };
        Ok(Self {
            workspace,
            scalar,
            scalar_workspace: None,
            sequential: matches!(equation, PredictionEquation::Sequential { .. }),
            token_quoted: false,
            output_quoted: false,
            completed_state_roots: None,
            completion_output_roots: 0,
        })
    }
    pub(super) fn finish_quote(&self) -> Result<(), IoError> {
        if self.token_quoted != self.scalar.is_some() || !self.output_quoted || self.completed_state_roots.is_none() {
            Err(IoError::Source)
        } else {
            Ok(())
        }
    }
    pub(in crate::composition::mlx::replicated_text) fn input_facts(
        &self,
    ) -> Option<OriginalPromptInputFacts> {
        self.scalar.map(|(_, _, facts)| facts)
    }
    pub(in crate::composition::mlx::replicated_text) fn retained_roots(&self) -> usize {
        usize::from(self.scalar.is_some())
    }
    pub(in crate::composition::mlx::replicated_text) fn completed_state_root_count(&self) -> Option<usize> {
        self.completed_state_roots
    }
    pub(in crate::composition::mlx::replicated_text) fn completed_output_root_count(&self) -> Option<usize> {
        self.output_quoted.then_some(self.completion_output_roots)
    }
    /// Exact H destinations are paid here before the final clone slot is born.
    /// Source graph/backing capacity is separately included in native admission.
    pub(in crate::composition::mlx::replicated_text) fn prepare(
        self,
        role: OriginalEmbeddedSpeculativeRole,
        plan: &InferenceSpanWorkspacePlan,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<PreparedPredictionIo, Error> {
        let funding = sources.metadata_funding();
        role.validate_plan(plan)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        role.validate_invocation(self.workspace.invocation())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if role.geometry() != self.workspace.geometry() {
            return Err(sources.retain_startup_error(IoError::Source));
        }
        self.finish_quote()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let parts = [
            size_of::<PreparedPredictionIo>(),
            size_of::<Result<PreparedPredictionIo, Error>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<Option<Array>>(),
            size_of::<Option<PreparedArrayClone>>(),
            size_of::<Result<(), IoError>>(),
            size_of::<[i32; 1]>(),
            size_of::<&WorkspaceMetadataFunding>(),
            size_of::<(
                &mut PreparedPredictionIo,
                u32,
                &OriginalEmbeddedSpeculativeRole,
                &OriginalScopeObserver,
            )>(),
            size_of::<(&PreparedPredictionIo, &mut dyn FnMut(&Array))>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .and_then(|n| {
                if let Some((_, _, facts)) = self.scalar {
                    n.checked_add(facts.control_bytes()?)?
                        .checked_add(PreparedArrayClone::control_bytes()?)?
                        .checked_add(Array::inspection_clone_handle_bytes())
                } else {
                    Some(n)
                }
            })
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let clone = if self.scalar.is_some() {
            Some(
                PreparedArrayClone::try_prepare_for_inspection()
                    .map_err(|cause| sources.retain_startup_error(cause))?,
            )
        } else {
            None
        };
        Ok(PreparedPredictionIo {
            scalar: None,
            clone,
            plan: self,
            spent: false,
            role,
            funding: funding.clone(),
        })
    }
}
impl WorkspacePredictionEquationTails for PredictionIoPlan {
    fn completed_state_roots(&mut self, count: usize, context: &WorkspaceContext) -> Result<(), eredu_nn::Error> {
        let frames=[size_of::<(&mut Self,usize,&WorkspaceContext)>(),size_of::<Result<(),eredu_nn::Error>>()];
        context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or(WorkspaceMetadataError::Overflow)?)?;
        if self.completed_state_roots.is_some() {return Err(context.metadata_source(IoError::Consumed));}
        self.completed_state_roots=Some(count);
        Ok(())
    }
    fn token(
        &mut self,
        id: u32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, eredu_nn::Error> {
        let parts = [
            size_of::<(&mut Self, u32, &WorkspaceContext)>(),
            size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
            size_of::<[i32; 2]>(),
        ];
        context.charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if self.token_quoted {
            return Err(context.metadata_source(IoError::Consumed));
        }
        let Some((expected, _, _)) = self.scalar else {
            return Err(context.metadata_source(IoError::Source));
        };
        if id != expected {
            return Err(context.metadata_source(IoError::Source));
        }
        self.token_quoted = true;
        // This existing source is backed by the exact eager I32 upload facts in
        // this plan. It is not an Initialize equation or a fabricated U32 cast.
        let value =
            WorkspaceTensor::existing(context.layout(&[1, 1], WorkspaceDtype::Int32)?, context)?;
        self.scalar_workspace = Some(value.clone());
        Ok(value)
    }
    fn visit_retained(&self, visit: &mut dyn FnMut(&WorkspaceTensor)) {
        if let Some(value) = &self.scalar_workspace {
            visit(value);
        }
    }
    fn readout(
        &mut self,
        output: &PredictionEquationOutput<WorkspaceTensor>,
        rows: &mut Vec<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        let parts = [
            size_of::<(
                &mut Self,
                &PredictionEquationOutput<WorkspaceTensor>,
                &mut Vec<WorkspaceTensor>,
                &WorkspaceContext,
            )>(),
            size_of::<[Index; 3]>(),
            size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
            size_of::<Result<(), eredu_nn::Error>>(),
        ];
        context.charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if self.output_quoted {
            return Err(context.metadata_source(IoError::Consumed));
        }
        self.output_quoted = true;
        self.completion_output_roots = match output {
            PredictionEquationOutput::Sequential { .. } => 2,
            PredictionEquationOutput::Fused(Some(_)) => 1,
            _ => 0,
        };
        match output {
            PredictionEquationOutput::Sequential { logits, .. } if self.sequential => {
                if logits.shape().len() != 3
                    || logits.shape()[0] != 1
                    || logits.shape()[1] != 1
                    || logits.shape()[2] <= 0
                {
                    return Err(context.metadata_source(IoError::Source));
                }
                context.reserve_metadata_vec(rows, 1)?;
                rows.push(logits.index(&[Index::Full, Index::At(0), Index::Full], context)?);
                Ok(())
            }
            PredictionEquationOutput::StateOnly if !self.sequential && self.scalar.is_none() => {
                Ok(())
            }
            // Fused row extraction happens lazily in the later existing consumer.
            PredictionEquationOutput::Fused(_) if !self.sequential && self.scalar.is_some() => {
                Ok(())
            }
            _ => Err(context.metadata_source(IoError::Source)),
        }
    }
}

pub(in crate::composition::mlx::replicated_text) struct PreparedPredictionIo {
    scalar: Option<Array>,
    clone: Option<PreparedArrayClone>,
    plan: PredictionIoPlan,
    spent: bool,
    role: OriginalEmbeddedSpeculativeRole,
    funding: WorkspaceMetadataFunding,
}
impl PreparedPredictionIo {
    pub(in crate::composition::mlx::replicated_text) fn token(
        &mut self,
        id: u32,
        role: &OriginalEmbeddedSpeculativeRole,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, Error> {
        let fail = |cause| retain_planning_error(cause, self.funding.clone());
        if !self.role.same_role(role) || self.spent {
            return Err(fail(IoError::Consumed));
        }
        let Some((expected, signed, _)) = self.plan.scalar else {
            return Err(fail(IoError::Source));
        };
        if expected != id {
            return Err(fail(IoError::Source));
        }
        let current = OriginalScopeObserver::require_current()
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
        if !current.same_scope(observer) {
            return Err(fail(IoError::Source));
        }
        // Spend before any native birth. Failure keeps the completed prefix and
        // original native failure under the enclosing active recovery owner.
        self.spent = true;
        self.scalar = Some(
            Array::try_from_original_prediction_ids(&[signed])
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?,
        );
        self.clone
            .as_mut()
            .ok_or_else(|| fail(IoError::Consumed))?
            .fill_in_original_scope(self.scalar.as_ref().expect("uploaded scalar"), observer)
            .map_err(|cause| retain_planning_error(cause, self.funding.clone()))
    }
    pub(in crate::composition::mlx::replicated_text) fn visit_retained(
        &self,
        visit: &mut dyn FnMut(&Array),
    ) {
        if let Some(value) = &self.scalar {
            visit(value);
        }
    }
}
