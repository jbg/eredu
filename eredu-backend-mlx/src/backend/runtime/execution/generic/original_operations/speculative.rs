//! One exact admitted AR/Embedded invocation uses the selected residency bank.
use eredu_runtime::{speculative::external_occurrence::ExternalInvocation, working_memory::OriginalExternalSpeculativeRole};
use super::*;
use crate::backend::nn::workspace::{AutoregressiveEquationRecipe, EmbeddedEquationRecipe, ResidentNativeRecipe};
use eredu_runtime::working_memory::{OriginalSpeculativePrefillSpan, OriginalSpeculativeRole, OriginalEmbeddedSpeculativeRole};

impl<U: 'static> OriginalOperationPlan<'_, U> {
    fn with_prepared_speculative_source(mut self) -> Result<Self, Error> {
        if let Some(source) = self.retained_sources {
            if source.has_original_foreground_source(&self.manager, self.window_sources.as_ref().ok_or_else(unknown)?) {
                self.foreground_disk_loan = Some(source.speculative_foreground().ok_or_else(unknown)?);
            }
        }
        Ok(self)
    }
    fn prepare_speculative_source(
        mut self, pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        let source = self.retained_sources.ok_or_else(unknown)?;
        let windows = self.window_sources.as_ref().ok_or_else(unknown)?;
        source.validate_original_policy(&self.manager, &self.identity, self.sources.as_slice())?;
        if source.has_original_foreground_source(&self.manager, windows) {
            let population = ResidencyPopulation::from_window_forwards(self.residency.ok_or_else(unknown)?.controller_units, windows.as_slice(), 1)?;
            self.foreground_disk_loan = Some(source.prepare_speculative_foreground(|| {
                storage::PreparedSpeculativeForegroundSource::new(&self.manager, pool, windows.as_slice(), self.sources.as_slice(), population, funding)?
                    .with_background(self.background.as_ref(), &self.manager, self.sources.as_slice(), windows.as_slice())
            })?);
        }
        Ok(self)
    }
    fn bind_speculative_recipe(
        &self,
        recipe: &mut AutoregressiveEquationRecipe,
    ) -> Result<(), Error> {
        recipe.with_native_recipe(|recipe|self.bind_role_recipe(recipe))
    }
    fn bind_role_recipe(&self, recipe:&mut ResidentNativeRecipe)->Result<(),Error> {
        let source = self.retained_sources.ok_or_else(unknown)?;
        source.validate_original_policy(&self.manager, &self.identity, self.sources.as_slice())?;
        let windows = self.window_sources.as_ref().ok_or_else(unknown)?;
        if !source.has_original_ready_host_source(&self.manager, windows)
            && !(self.foreground_disk_plan().is_some() && source.has_original_foreground_source(&self.manager, windows)) {
            return Err(unknown());
        }
        self.bind_neural_recipe(recipe)?;
        source.bind_native_host_copies(recipe)
    }
    fn for_speculative_span(
        mut self,
        recipe: &ResidentNativeRecipe,
    ) -> Result<Self, Error> {
        self = self.with_prepared_speculative_source()?;
        self = self.with_neural_recipe(Some(recipe))?;
        let source = self.retained_sources.ok_or_else(unknown)?;
        let windows = self.window_sources.as_ref().ok_or_else(unknown)?;
        source.validate_original_policy(&self.manager, &self.identity, self.sources.as_slice())?;
        if !source.has_original_ready_host_source(&self.manager, windows)
            && !(self.foreground_disk_plan().is_some() && source.has_original_foreground_source(&self.manager, windows)) {
            return Err(unknown());
        }
        // Each actual callback/invocation owns one finite bank. This is the
        // same per-window population worker, with one actual forward per bank.
        self.residency = Some(ResidencyPopulation::from_window_forwards(
            self.residency.ok_or_else(unknown)?.controller_units,
            windows.as_slice(),
            1,
        )?);
        self.units = self.sources.len();
        self.pending = self.units;
        self.scopes = 1;
        self.neural.submissions = self.neural.per_forward;
        // with_neural_recipe above authenticated the exact complete reduction,
        // including the selected source and all group/materialization boundaries.
        // Restricting that already proved population to one recorded forward
        // preserves its fit; an unvalidated geometry never reaches this step.
        self.neural_fit = NeuralProducerFit::Recipe(
            NeuralProducerFit::pending(self.neural)
                .ok_or_else(overflow)?
                .requirement(),
        );
        Ok(self)
    }
    fn speculative_span_control_bytes(self, recipe: &ResidentNativeRecipe) -> Result<u64, Error> {
        let value = self.for_speculative_span(recipe)
            .map_err(|cause| cause.at_speculative_stage("selected residency recipe validation"))?;
        value.control_bytes()
            .and_then(|bytes| bytes.checked_add(Self::speculative_extra_control_bytes()?))
            .ok_or_else(|| unknown().at_speculative_stage("selected residency control bound"))
    }
    fn prepare_speculative_bank(
        self, recipe:&AutoregressiveEquationRecipe, role:OriginalSpeculativeRole,
        span:Option<&OriginalSpeculativePrefillSpan>, scope:&safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    )->Result<Option<SpeculativeNeuralOwner>,Error> {
        let tagged=SpeculativeOperationRole::Autoregressive(role);
        let SpeculativeOperationRole::Autoregressive(role)=&tagged else {unreachable!()};
        let validation=(|| {
            role.validate_plan(recipe.plan()).map_err(memory)?;
            role.validate_invocation(recipe.invocation()).map_err(memory)?;
            if let Some(span)=span {
                span.validate_plan(recipe.plan()).map_err(memory)?;
                if !span.role().same_role(role) {return Err(identity());}
            } else if recipe.records().len()!=1 {return Err(identity());}
            Ok(())
        })();
        validation.map_err(|cause|neural::boundary_error(cause,&tagged))?;
        self.prepare_role_bank(recipe.native_recipe(),tagged,span,scope,partition)
    }
    fn prepare_embedded_bank(
        self, recipe:&EmbeddedEquationRecipe, role:OriginalEmbeddedSpeculativeRole,
        scope:&safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    )->Result<Option<SpeculativeNeuralOwner>,Error> {
        let tagged=SpeculativeOperationRole::Embedded(role);
        let SpeculativeOperationRole::Embedded(role)=&tagged else {unreachable!()};
        let validation=(|| {
            role.validate_plan(recipe.plan()).map_err(memory)?;
            role.validate_invocation(recipe.workspace().invocation()).map_err(memory)?;
            if role.geometry()!=recipe.workspace().geometry() {return Err(identity());}
            Ok(())
        })();
        validation.map_err(|cause|neural::boundary_error(cause,&tagged))?;
        self.prepare_role_bank(recipe.native_recipe(),tagged,None,scope,partition)
    }
    fn prepare_external_bank(
        self, recipe: &ResidentNativeRecipe, invocation: ExternalInvocation,
        role: OriginalExternalSpeculativeRole, scope: &safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        let tagged = SpeculativeOperationRole::External(role);
        let validation = (|| {
            let SpeculativeOperationRole::External(role) = &tagged else { unreachable!() };
            role.validate_plan(recipe.plan()).map_err(memory)?;
            role.validate_invocation(invocation).map_err(memory)?;
            if recipe.records().len() != 1 { return Err(identity()); }
            Ok(())
        })();
        validation.map_err(|cause| neural::boundary_error(cause, &tagged))?;
        self.prepare_role_bank(recipe, tagged, None, scope,partition)
    }
    fn prepare_role_bank(
        self, recipe:&ResidentNativeRecipe, role:SpeculativeOperationRole,
        span:Option<&OriginalSpeculativePrefillSpan>, scope:&safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    )->Result<Option<SpeculativeNeuralOwner>,Error> {
        let result=(|| {
            let value = self.for_speculative_span(recipe)?;
            let bytes = value
                .control_bytes()
                .ok_or_else(unknown)?
                .checked_add(Self::speculative_extra_control_bytes().ok_or_else(unknown)?)
                .ok_or_else(overflow)?;
            let observer = safemlx::OriginalScopeObserver::require_current()?;
            if !observer.belongs_to(scope) {
                return Err(identity());
            }
            let previous = value.slot.take();
            let busy = previous
                .as_ref()
                .is_some_and(|view| view.value.strong_count() != 0);
            value.slot.set(previous);
            if busy {
                return Err(Error::PrefillScopeReentrant);
            }
            match span {
                Some(span) => span.claim_neural_bank(bytes),
                None => role.claim_neural_bank(bytes),
            }
            .map_err(memory)?;
            let custody: eredu_runtime::working_memory::OriginalOperationMetadataCustody =
                role.budget_custody().into();
            let disk_source = match span {
                Some(span) => span.take_host_source_constructions(),
                None => role.take_host_source_constructions(),
            }.map_err(memory)?;
            let disk_source = match partition {Some(partition)=>partition(disk_source)?,None=>disk_source};
            let disk_source = match (value.foreground_disk_plan(), disk_source) {
                (Some(plan), Some(bank)) if bank.matches_facts(plan.source_facts().ok_or_else(unknown)?) =>
                    Some(storage::ForegroundDiskSourceTicket::Source { bank, custody: role.budget_custody().into() }),
                (None, None) => None,
                _ => return Err(identity()),
            };
            let value=value.prepare_accepted_bank(observer,custody,
                OperationRequest::Speculative(role.clone()),OperationControls::Speculative(role.clone()),disk_source)?;
            Ok(Some(SpeculativeNeuralOwner {value,role:role.clone()}))
        })();
        result.map_err(|cause| neural::boundary_error(cause, &role))
    }
    fn speculative_extra_control_bytes() -> Option<u64> {
        u64::try_from(
            [
                size_of::<SpeculativeNeuralOwner>(),
                size_of::<Result<Option<SpeculativeNeuralOwner>, Error>>(),
                size_of::<SpeculativeOperationRole>(),
                size_of::<Option<&OriginalSpeculativePrefillSpan>>(),
                size_of::<OperationControls>(),
                size_of::<eredu_runtime::working_memory::OriginalHostSourceCustody>(),
                size_of::<Option<eredu_runtime::working_memory::OriginalHostSourceBank>>(),
                size_of::<Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>, eredu_runtime::working_memory::WorkingMemoryError>>(),
                size_of::<Option<storage::ForegroundDiskSourceTicket>>(),
                size_of::<Option<SpeculativeSourcePartition<'_>>>(),
                size_of::<Result<Option<eredu_runtime::working_memory::OriginalHostSourceBank>,Error>>(),
            ]
            .into_iter()
            .try_fold(0usize, usize::checked_add)?,
        )
        .ok()?
        .checked_add(
            u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
                neural::NeuralBoundaryFailure<SpeculativeOperationRole>,
            >()?)
            .ok()?,
        )
    }
}
impl<U: 'static> SelectedOriginalOperationPlan<'_, U> {
    pub(crate) fn prepare_speculative_source(
        self, pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        match self {
            Self::Bounded(plan) => plan.prepare_speculative_source(pool, funding).map(Self::Bounded),
            Self::Resident(plan) => Ok(Self::Resident(plan)),
        }
    }
    pub(crate) fn speculative_source_facts(&self) -> Result<Option<eredu_runtime::working_memory::HostSourceConstructionFacts>, Error> {
        match self {
            Self::Bounded(plan) => plan.foreground_disk_plan().map(|source| source.source_facts().ok_or_else(unknown)).transpose(),
            Self::Resident(_) => Ok(None),
        }
    }

    pub(crate) fn bind_speculative_recipe(
        &self,
        recipe: &mut AutoregressiveEquationRecipe,
    ) -> Result<(), Error> {
        match self {
            Self::Bounded(plan) => plan.bind_speculative_recipe(recipe),
            Self::Resident(plan) => plan.bind_speculative_recipe(recipe),
        }
    }
    pub(crate) fn speculative_control_bytes(
        self,
        recipe: &AutoregressiveEquationRecipe,
    ) -> Result<u64, Error> {
        match self {
            Self::Bounded(plan) => {
                let forwards = recipe.records().len();
                plan.speculative_span_control_bytes(recipe.native_recipe())?
                    .checked_mul(u64::try_from(forwards).map_err(|_| overflow())?)
                    .ok_or_else(overflow)
            }
            Self::Resident(plan) => {
                if recipe.invocation().execution_pass() == eredu_runtime::ExpertPass::Prefill {
                    plan.speculative_span_control_bytes(recipe)
                        .and_then(|bytes| bytes.checked_mul(u64::try_from(recipe.records().len()).ok()?))
                        .ok_or_else(|| unknown().at_speculative_stage("resident span control bound"))
                } else {
                    plan.speculative_control_bytes(recipe)
                        .ok_or_else(|| unknown().at_speculative_stage("resident invocation control bound"))
                }
            }
        }
    }
    pub(crate) fn bind_external_recipe(&self, recipe: &mut ResidentNativeRecipe) -> Result<(), Error> {
        if recipe.records().len() != 1 { return Err(identity()); }
        match self {
            Self::Bounded(plan) => plan.bind_role_recipe(recipe),
            Self::Resident(plan) => plan.bind_external_recipe(recipe),
        }
    }
    pub(crate) fn external_control_bytes(self, recipe: &ResidentNativeRecipe) -> Result<u64, Error> {
        if recipe.records().len() != 1 { return Err(identity()); }
        match self {
            Self::Bounded(plan) => plan.speculative_span_control_bytes(recipe),
            Self::Resident(plan) => plan.external_control_bytes(recipe).ok_or_else(unknown),
        }
    }
    pub(crate) fn prepare_external(
        self, recipe: &ResidentNativeRecipe, invocation: ExternalInvocation,
        role: OriginalExternalSpeculativeRole, scope: &safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        match self {
            Self::Bounded(plan) => plan.prepare_external_bank(recipe, invocation, role, scope,partition),
            Self::Resident(plan) => plan.prepare_external(recipe, invocation, role, scope,partition),
        }
    }
    pub(crate) fn bind_embedded_recipe(&self,recipe:&mut EmbeddedEquationRecipe)->Result<(),Error> {
        match self {
            Self::Bounded(plan)=>recipe.with_native_recipe(|recipe|plan.bind_role_recipe(recipe)),
            Self::Resident(plan)=>plan.bind_embedded_recipe(recipe),
        }
    }
    pub(crate) fn embedded_control_bytes(self,recipe:&EmbeddedEquationRecipe)->Result<u64,Error> {
        match self {
            Self::Bounded(plan)=>plan.speculative_span_control_bytes(recipe.native_recipe()),
            Self::Resident(plan)=>plan.embedded_control_bytes(recipe).ok_or_else(unknown),
        }
    }
    pub(crate) fn prepare_embedded(self,recipe:&EmbeddedEquationRecipe,
        role:OriginalEmbeddedSpeculativeRole,scope:&safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    )->Result<Option<SpeculativeNeuralOwner>,Error> {
        match self {
            Self::Bounded(plan)=>plan.prepare_embedded_bank(recipe,role,scope,partition),
            Self::Resident(plan)=>plan.prepare_embedded(recipe,role,scope,partition),
        }
    }
    pub(crate) fn prepare_speculative(
        self,
        recipe: &AutoregressiveEquationRecipe,
        role: OriginalSpeculativeRole,
        scope: &safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        match self {
            Self::Bounded(plan) => plan.prepare_speculative_bank(recipe, role, None, scope,partition),
            Self::Resident(plan) => plan.prepare_speculative(recipe, role, scope,partition),
        }
    }
    pub(crate) fn prepare_speculative_span(
        self,
        recipe: &AutoregressiveEquationRecipe,
        span: &OriginalSpeculativePrefillSpan,
        scope: &safemlx::SubmissionScope,
        partition:Option<SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<SpeculativeNeuralOwner>, Error> {
        match self {
            Self::Bounded(plan) => {
                plan.prepare_speculative_bank(recipe, span.role().clone(), Some(span), scope,partition)
            }
            Self::Resident(plan) => plan.prepare_speculative_span(recipe, span, scope,partition),
        }
    }
}
