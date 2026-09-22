//! Shared original Host-copy and replaced-unit constructor populations.
use super::super::host_copies::PreparedSourceCopies;
use super::*;
use crate::backend::runtime::execution::generic::LayerwiseWorkspace;
use crate::backend::runtime::residency::manager::WindowPopulation;
use safemlx::OperationEvent;

impl SpeculativeNumericalRecipe {
    /// Join the independently scoped copy/aggregate producers only after the
    /// numerical equation population is complete. Their constructors share the
    /// graph arena, but are not entries in the equation's Eval tape.
    pub(crate) fn with_indexed_source(
        self,
        residency: &crate::backend::runtime::residency::parameter_bank::IndexedResidencyPlan,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable transfer source differs from its actual numerical population"
            ))
        };
        context.charge_metadata(std::mem::size_of::<(
            Self,
            &crate::backend::runtime::residency::parameter_bank::IndexedResidencyPlan,
            &WorkspaceContext,
            PreparedSourceCopies,
            Result<Self, Error>,
            &crate::backend::runtime::residency::manager::SupplementaryResidencySource,
            WindowPopulation,
            usize,
            Result<PreparedSourceCopies, Error>,
        )>())?;
        context.charge_metadata(
            PreparedSourceCopies::inspection_control_bytes().ok_or_else(invalid)?,
        )?;
        let source = residency
            .with_native_copy_source(|source, window, calls| {
                PreparedSourceCopies::inspect(source, window, calls)
            })
            .map_err(|cause| context.metadata_source(cause))?;
        self.with_prepared_source_copies(source, context)?
            .with_materialized_storage(
                residency
                    .materialized_population()
                    .map_err(|cause| context.metadata_source(cause))?,
                context,
            )
    }

    /// Child producer arenas have separately prepaid metadata. Only their
    /// actual native backing births join this enclosing buffer source.
    pub(crate) fn with_materialized_storage(
        mut self,
        population: crate::backend::runtime::residency::manager::ForegroundMaterializationPopulation,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let overflow =
            || context.metadata_source(eredu_runtime::working_memory::WorkingMemoryError::Overflow);
        self.storage.mutable_bytes = self
            .storage
            .mutable_bytes
            .checked_add(population.bytes)
            .ok_or_else(overflow)?;
        self.storage.maximum_births = self
            .storage
            .maximum_births
            .checked_add(population.births)
            .ok_or_else(overflow)?;
        Ok(self)
    }

    /// Extend the same native source reducer with the genuine prepared recipe
    /// leaf loads and final Host store. These are descriptive populations; the
    /// caller still authenticates every leaf and supplies accepted role banks.
    pub(crate) fn with_prepared_recipe_transfers<I>(
        self,
        leaves: I,
        output_rank: usize,
        output_dtype: safemlx::Dtype,
        output_capacity: u64,
        context: &WorkspaceContext,
    ) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (usize, safemlx::Dtype, u64)>,
    {
        context.charge_metadata(std::mem::size_of::<(
            I,
            I::IntoIter,
            Self,
            PreparedSourceCopies,
            Result<Self, Error>,
            (usize, safemlx::Dtype, u64),
        )>())?;
        let source = PreparedSourceCopies::inspect_prepared_recipe(
            leaves,
            output_rank,
            output_dtype,
            output_capacity,
        )
        .map_err(|cause| context.metadata_source(cause))?;
        // The prepared recipe explicitly submits its equation output before
        // the separately completed store. The retained trace alone has no
        // child-final frontier; include this actual caller before reducing the
        // same Graph/Record capacities with its leaf and store sources.
        let mut recipe = self;
        recipe.completion.nested_completions = recipe
            .completion
            .nested_completions
            .checked_add(1)
            .ok_or_else(|| {
                context.metadata_source(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
            })?;
        recipe.with_prepared_source_copies(source, context)
    }

    pub(in super::super) fn with_prepared_source_copies(
        self,
        source: PreparedSourceCopies,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "retained transfer source differs from its actual numerical population"
            ))
        };
        context.charge_metadata(std::mem::size_of::<(
            Self,
            PreparedSourceCopies,
            ResidentCompletionRecipe,
            ResidentDispatchPopulation,
            safemlx::OperationEvalTraversalLimits,
            Result<Self, Error>,
            [usize; 8],
        )>())?;
        let mut result = self;
        let mut completion = result.completion;
        let copies = source.copies;
        let transfers = source.transfers;
        let count = copies.per_forward;
        let mut limits = completion.traversal.limits();
        limits.arrays = limits
            .arrays
            .checked_add(
                count
                    .checked_add(source.constructor_primitives)
                    .ok_or_else(invalid)?,
            )
            .ok_or_else(invalid)?;
        limits.input_edges = limits
            .input_edges
            .checked_add(source.constructor_primitives)
            .ok_or_else(invalid)?;
        completion.traversal = OperationEvent::eval_traversal_layout(limits).ok_or_else(invalid)?;
        let graph = completion.graph;
        completion.graph = OperationEvent::resident_graph_layout_with_shells(
            graph
                .primitives()
                .checked_add(source.constructor_primitives)
                .ok_or_else(invalid)?,
            graph.seeds().checked_add(count).ok_or_else(invalid)?,
            graph.maximum_rank().max(source.rank),
            graph.maximum_operands().max(4),
            graph
                .additional_shells()
                .checked_add(transfers.binding_shells)
                .ok_or_else(invalid)?,
        )
        .ok_or_else(invalid)?;
        let (dispatch, worker_controls) = source
            .expand_equation_dispatch(
                completion.dispatch.ok_or_else(invalid)?,
                limits,
                completion.graph.maximum_operands(),
            )
            .map_err(|cause| context.metadata_source(cause))?;
        completion.dispatch = Some(dispatch);
        // General nested completions use this configured bank's array and
        // edge capacities even when their actual root list is shorter. Bind
        // the selected source roots before pricing that same native bank.
        completion.nested_root_capacity = completion.nested_root_capacity.max(transfers.roots);
        // Equation completions retain their own finite root distribution.
        // Transfers have separate exact Eval/Wait producers in the same arena.
        let mut graph =
            graph_capacity::ResidentGraphStorage::for_completion(completion).ok_or_else(invalid)?;
        graph
            .include_source_copies(copies, transfers, dispatch)
            .ok_or_else(invalid)?;
        let records = record_capacity::ResidentRecordStorage::for_completion_sources(
            completion,
            0,
            Some(source),
        )
        .ok_or_else(invalid)?;
        let equation_frontiers = if dispatch.cpu_entries == 0 {
            1
        } else {
            completion
                .nested_completions
                .checked_add(1)
                .ok_or_else(invalid)?
        };
        completion.nested_completions = completion
            .nested_completions
            .checked_add(count)
            .and_then(|n| n.checked_add(transfers.per_forward))
            .ok_or_else(invalid)?;
        result.completion = completion;
        result.storage.mutable_bytes = result
            .storage
            .mutable_bytes
            .checked_add(transfers.bytes)
            .ok_or_else(invalid)?;
        result.storage.maximum_births = result
            .storage
            .maximum_births
            .checked_add(count)
            .ok_or_else(invalid)?;
        result.graph_capacity =
            usize::try_from(graph.full_capacity.ok_or_else(invalid)?).map_err(|_| invalid())?;
        result.record_capacity =
            usize::try_from(records.full_capacity.ok_or_else(invalid)?).map_err(|_| invalid())?;
        result.kernels = dispatch
            .kernel_attempts
            .checked_mul(equation_frontiers)
            .and_then(|n| n.checked_add(copies.dispatch.kernel_attempts.checked_mul(count)?))
            .and_then(|n| {
                n.checked_add(
                    copies
                        .aggregate_dispatch
                        .kernel_attempts
                        .checked_mul(transfers.per_forward)?,
                )
            })
            .ok_or_else(invalid)?;
        result.controls = result
            .controls
            .checked_add(u64::try_from(source.controls).map_err(|_| invalid())?)
            .and_then(|n| n.checked_add(graph.control_bytes()?))
            .and_then(|n| n.checked_add(u64::try_from(worker_controls).ok()?))
            .ok_or_else(invalid)?;
        Ok(result)
    }

    /// The source is the same retained selected parameter/copy inventory used
    /// by the operation bank. Windows describe actual finite warm/missing
    /// attempts; no text-forward geometry or synthetic parameter provider is
    /// introduced for a realtime frame.
    pub(crate) fn with_layerwise_source(
        self,
        source: &LayerwiseWorkspace,
        windows: &[WindowPopulation],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("retained layerwise source is incomplete"));
        context.charge_metadata(
            PreparedSourceCopies::inspection_control_bytes().ok_or_else(invalid)?,
        )?;
        let copies = PreparedSourceCopies::inspect_layerwise(source, windows)
            .map_err(|cause| context.metadata_source(cause))?;
        let (constructors, query) = super::super::parameter_construction::constructor_source(
            source,
            context.metadata_funding().as_ref(),
        )
        .map_err(|cause| context.metadata_source(cause))?;
        let (additional, controls) = constructors
            .native_source(if context.metadata_funding().is_some() {
                0
            } else {
                query
            })
            .map_err(|cause| context.metadata_source(cause))?;
        self.with_parameter_source_parts(copies, constructors, additional, controls, context)
    }

    /// One actual main-unit constructor and its selected source closure. The
    /// ordinal census comes from the installed manager, not supplementary IDs.
    pub(crate) fn with_selected_layerwise_source(
        self,
        source: &LayerwiseWorkspace,
        window: WindowPopulation,
        constructors: crate::backend::runtime::execution::generic::ParameterConstructors,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("selected parameter source is incomplete"));
        context.charge_metadata(
            PreparedSourceCopies::inspection_control_bytes().ok_or_else(invalid)?,
        )?;
        let copies = PreparedSourceCopies::inspect_layerwise(source, std::slice::from_ref(&window))
            .map_err(|cause| context.metadata_source(cause))?;
        let (additional, controls) = constructors
            .native_source(0)
            .map_err(|cause| context.metadata_source(cause))?;
        self.with_parameter_source_parts(copies, constructors, additional, controls, context)
    }
    fn with_parameter_source_parts(
        self,
        copies: PreparedSourceCopies,
        constructors: crate::backend::runtime::execution::generic::ParameterConstructors,
        additional: safemlx::ResidentGraphLayout,
        controls: usize,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid =
            || context.metadata_error(format_args!("retained layerwise source is incomplete"));
        let mut result = self;
        let graph = result.completion.graph;
        result.completion.graph = OperationEvent::resident_graph_layout_with_shells(
            graph
                .primitives()
                .checked_add(additional.primitives())
                .ok_or_else(invalid)?,
            graph
                .seeds()
                .checked_add(additional.seeds())
                .ok_or_else(invalid)?,
            graph.maximum_rank().max(additional.maximum_rank()),
            graph.maximum_operands().max(additional.maximum_operands()),
            graph.additional_shells(),
        )
        .ok_or_else(invalid)?;
        result.storage.mutable_bytes = result
            .storage
            .mutable_bytes
            .checked_add(constructors.scalar_bytes)
            .ok_or_else(invalid)?;
        result.storage.maximum_births = result
            .storage
            .maximum_births
            .checked_add(constructors.slots)
            .ok_or_else(invalid)?;
        result.controls = result
            .controls
            .checked_add(u64::try_from(controls).map_err(|_| invalid())?)
            .ok_or_else(invalid)?;
        // Constructors are replaced before Eval. Only the common copy reducer
        // extends the real traversal, native copy workers and completion bank.
        result.with_prepared_source_copies(copies, context)
    }
}
