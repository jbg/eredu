//! Typed source identity adapters around the one native scope/recovery worker.
use super::*;
use crate::backend::nn::workspace::{ResidentNativeRecipe, ResidentSpanRecipe};
use crate::backend::runtime::distributed::topology::original_source::control::SpeculativeModelRole;
use eredu_runtime::{
    speculative::{
        embedded_occurrence::EmbeddedInvocationWorkspace, external_occurrence::ExternalInvocation,
    },
    working_memory::{
        InferenceSpanWorkspacePlan, OriginalExternalSpeculativeRole, WorkingMemoryError,
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EquationIdentity {
    Embedded(EmbeddedInvocationWorkspace),
    External(ExternalInvocation),
}
/// Closed backend adapters preserve each actual source/role kind.
pub(crate) trait ModelEquation {
    fn plan(&self) -> &InferenceSpanWorkspacePlan;
    fn native_recipe(&self) -> &ResidentNativeRecipe;
    fn identity(&self) -> EquationIdentity;
    fn record(&self) -> &ResidentSpanRecipe {
        &self.native_recipe().records()[0]
    }
    fn equation_domains(
        &self,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), Error>;
    fn validation_control_bytes(&self) -> Result<u64, Error>;
}
impl ModelEquation for EmbeddedEquationRecipe {
    fn plan(&self) -> &InferenceSpanWorkspacePlan {
        self.plan()
    }
    fn native_recipe(&self) -> &ResidentNativeRecipe {
        self.native_recipe()
    }
    fn identity(&self) -> EquationIdentity {
        EquationIdentity::Embedded(self.workspace())
    }
    fn equation_domains(
        &self,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), Error> {
        self.equation_domains(preparation_graph_bytes)
    }
    fn validation_control_bytes(&self) -> Result<u64, Error> {
        TokenValidationIngress::embedded_control_bytes(self)
    }
}

/// Actual external invocation plus the same single-equation native reducer.
pub(crate) struct ExternalEquationRecipe {
    invocation: ExternalInvocation,
    recipe: ResidentNativeRecipe,
}
impl ExternalEquationRecipe {
    pub(crate) fn new(
        invocation: ExternalInvocation,
        recipe: ResidentNativeRecipe,
    ) -> Result<Self, Error> {
        if recipe.plan().geometry() != invocation.geometry() || recipe.records().len() != 1 {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(Self { invocation, recipe })
    }
    pub(crate) fn invocation(&self) -> ExternalInvocation {
        self.invocation
    }
    pub(crate) fn plan(&self) -> &InferenceSpanWorkspacePlan {
        self.recipe.plan()
    }
    pub(crate) fn native_recipe(&self) -> &ResidentNativeRecipe {
        &self.recipe
    }
    pub(crate) fn with_native_recipe<T>(
        &mut self,
        run: impl FnOnce(&mut ResidentNativeRecipe) -> T,
    ) -> T {
        run(&mut self.recipe)
    }
}
impl ModelEquation for ExternalEquationRecipe {
    fn plan(&self) -> &InferenceSpanWorkspacePlan {
        self.plan()
    }
    fn native_recipe(&self) -> &ResidentNativeRecipe {
        self.native_recipe()
    }
    fn identity(&self) -> EquationIdentity {
        EquationIdentity::External(self.invocation)
    }
    fn equation_domains(
        &self,
        preparation_graph_bytes: usize,
    ) -> Result<(ResidentCompletionRecipe, usize, usize, usize, u64), Error> {
        self.recipe
            .external_equation_domains(preparation_graph_bytes)
    }
    fn validation_control_bytes(&self) -> Result<u64, Error> {
        TokenValidationIngress::external_control_bytes(&self.recipe)
    }
}

/// Only the concrete role adapters below supply authority to the shared worker.
pub(crate) trait ModelRole {
    type Equation: ModelEquation;
    fn model_role(&self) -> SpeculativeModelRole;
    fn validate_equation(&self, recipe: &Self::Equation) -> Result<(), Error>;
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody;
    fn physical_bytes(&self) -> u64;
    fn graph_bytes(&self) -> u64;
    fn record_bytes(&self) -> u64;
    fn prepare_validations(&self, recipe: &Self::Equation)
    -> Result<TokenValidationIngress, Error>;
}
impl ModelRole for OriginalEmbeddedSpeculativeRole {
    type Equation = EmbeddedEquationRecipe;
    fn model_role(&self) -> SpeculativeModelRole {
        SpeculativeModelRole::Embedded(self.clone())
    }
    fn validate_equation(&self, recipe: &Self::Equation) -> Result<(), Error> {
        self.validate_plan(recipe.plan())
            .and_then(|_| self.validate_invocation(recipe.workspace().invocation()))
            .map_err(Error::PrefillControl)
    }
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        self.budget_custody()
    }
    fn physical_bytes(&self) -> u64 {
        self.physical_bytes()
    }
    fn graph_bytes(&self) -> u64 {
        self.graph_bytes()
    }
    fn record_bytes(&self) -> u64 {
        self.record_bytes()
    }
    fn prepare_validations(
        &self,
        recipe: &Self::Equation,
    ) -> Result<TokenValidationIngress, Error> {
        TokenValidationIngress::prepare_embedded(recipe, self)
    }
}
impl ModelRole for OriginalExternalSpeculativeRole {
    type Equation = ExternalEquationRecipe;
    fn model_role(&self) -> SpeculativeModelRole {
        SpeculativeModelRole::External(self.clone())
    }
    fn validate_equation(&self, recipe: &Self::Equation) -> Result<(), Error> {
        self.validate_plan(recipe.plan()).map_err(|cause| {
            Error::PrefillControl(cause).at_speculative_stage("external equation plan")
        })?;
        if self.invocation() != recipe.invocation {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
                .at_speculative_stage("external equation invocation"));
        }
        Ok(())
    }
    fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        self.budget_custody()
    }
    fn physical_bytes(&self) -> u64 {
        self.physical_bytes()
    }
    fn graph_bytes(&self) -> u64 {
        self.graph_bytes()
    }
    fn record_bytes(&self) -> u64 {
        self.record_bytes()
    }
    fn prepare_validations(
        &self,
        recipe: &Self::Equation,
    ) -> Result<TokenValidationIngress, Error> {
        self.validate_equation(recipe)?;
        TokenValidationIngress::prepare_external(recipe.native_recipe(), self)
    }
}
