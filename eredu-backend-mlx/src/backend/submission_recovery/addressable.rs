//! One accepted addressable occurrence runs its ordinary provider callback in
//! the shared native role worker. Numerical quotation creates no source bank.
use super::{
    native_role::{self, NativeRoleCapacity},
    prefill::nested::NestedRootCompletion,
};
use crate::{
    backend::{
        error::Error,
        nn::{tensor::PreparedTokenChild, workspace::{AddressableQuote,AddressableQuoteRef}},
        runtime::{
            execution::generic::{
                OriginalSelectedResidencyAccess, PreparedSelectedResidencyAccess,
            },
            residency::{
                manager::ForegroundDiskSourceCapacity,
                parameter_bank::{
                    IndexedBankSource, IndexedConstructorPartitions,
                    OriginalIndexedResidencyFactory,
                },
            },
        },
    },
    MlxTensor,
};
use eredu_nn::{
    workspace::{HostMetadataFunding, HostMetadataFundingError},
    TensorParallelGroupedOutput,
};
use eredu_runtime::{expert::IndexedInvocationRequest, working_memory::OriginalOperationMetadataCustody};
use safemlx::{
    Array, OriginalScopeObserver, PreparedArrayClone, PreparedInputRuntime, PreparedStreamCopy,
    Stream, StreamCopyPlan,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Debug)]
struct Custody {
    raw: OriginalOperationMetadataCustody,
    funding: HostMetadataFunding,
}
struct Region {
    inputs: [Array; 4],
    completion: RefCell<NestedRootCompletion>,
    collector: PreparedTokenChild,
    quote: AddressableQuoteRef,
    access: PreparedSelectedResidencyAccess,
    constructors: RefCell<Option<IndexedConstructorPartitions>>,
    reads: Option<ForegroundDiskSourceCapacity>,
    runtime: PreparedInputRuntime,
    stream: PreparedStreamCopy<Custody>,
    custody: Custody,
}
fn identity() -> Error { mismatch("addressable child invocation") }
fn mismatch(stage: &'static str) -> Error {
    Error::OriginalSourceContract {
        stage,
        cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
    }
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
type Output = TensorParallelGroupedOutput<MlxTensor>;
type Callback<'a> = dyn FnMut(OriginalIndexedResidencyFactory) -> Result<Output, Error> + 'a;

fn entry_control_bytes()->Option<usize> {
    let frames=[
            size_of::<Region>(),
            size_of::<Custody>(),
            size_of::<&mut Callback<'_>>(),
            size_of::<[&Array; 4]>(),
            size_of::<[Array; 4]>(),
            size_of::<[PreparedArrayClone; 4]>(),
            size_of::<Result<Output, Error>>(),
            size_of::<Result<Result<Output, Error>, eredu_core::BackendFailure>>(),
            PreparedInputRuntime::inspection_alias_control_bytes(),
            4usize
                .checked_mul(
                    PreparedArrayClone::control_bytes()
                        .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
                        ?,
                )
                ?,
            safemlx::OperationEvent::nested_completion_control_bytes::<4>()?,
            4usize
                .checked_mul(
                    safemlx::OperationEvent::traversal_leaf_control_bytes()?,
                )
                ?,
        ];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
/// Exact shared native wrapper population. Constructor partition directories
/// and the request's shared reader program retain separate source declarations.
pub(crate) fn control_bytes(quote:&AddressableQuote)->Result<usize,Error> {
    envelope_control_bytes(quote, quote.numerical, quote.capacity)
}
pub(crate) fn envelope_control_bytes(quote:&AddressableQuote,
    recipe:crate::backend::nn::workspace::SpeculativeNumericalRecipe,
    capacity:crate::backend::nn::workspace::BoundaryStageCapacity)->Result<usize,Error> {
    let capacity=NativeRoleCapacity{graph:capacity.graph,records:capacity.records,backing:capacity.backing};
    let native=native_role::control_bytes::<Region,Custody>(capacity,
        Some(safemlx::PreparedPipelineCachePlan::new(recipe.kernels)))
        .map_err(|cause|Error::Neural(quote.identity().funding().metadata_source(cause)))?;
    let frames=[entry_control_bytes().ok_or_else(overflow)?,
        usize::try_from(recipe.controls).map_err(|_|overflow())?,
        NestedRootCompletion::control_bytes(quote.outputs.len(),recipe.completion.validation_roots).ok_or_else(overflow)?,
        PreparedTokenChild::control_bytes(recipe.completion).ok_or_else(overflow)?,
        StreamCopyPlan::<Custody>::constructor_storage_bytes()
            .map_err(|cause|Error::Neural(quote.identity().funding().metadata_source(cause)))?,
        quote.residency.invocation_control_bytes()?,
        native,native_role::callback_control_bytes::<Output,Error>(size_of::<&mut Callback<'_>>()).ok_or_else(overflow)?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or_else(overflow)
}

/// Constructor partitions and the shared reader capacity are consumed/borrowed
/// from this request's previously accepted source program. No byte count can
/// substitute for either authority.
pub(crate) fn run_region(
    quote: AddressableQuoteRef,
    source: &IndexedBankSource,
    request: IndexedInvocationRequest<'_, MlxTensor>,
    access: &OriginalSelectedResidencyAccess,
    constructors: IndexedConstructorPartitions,
    reads: Option<ForegroundDiskSourceCapacity>,
    runtime: &PreparedInputRuntime,
    budget: &safemlx::OriginalBufferBudget,
    metadata: &OriginalOperationMetadataCustody,
    funding: &HostMetadataFunding,
    timeout: Option<std::time::Duration>,
    stream: &Stream,
    run: &mut Callback<'_>,
) -> Result<Output, Error> {
    funding.reserve_metadata(entry_control_bytes().ok_or_else(overflow)?)
        .map_err(Error::WorkspacePlanning)?;
    funding.reserve_metadata(usize::try_from(quote.numerical.controls).map_err(|_|overflow())?)
        .map_err(Error::WorkspacePlanning)?;
    if !source.same_binding(quote.identity().source()) {
        return Err(mismatch("addressable bank binding"));
    }
    if request.declaration != quote.declaration.as_view() {
        return Err(mismatch("addressable invocation declaration"));
    }
    let originals = [
        request.input.as_array(),
        request.routes.group_indices().as_array(),
        request.routes.selected_scores().as_array(),
        request.routes.coefficients().as_array(),
    ];
    if quote.inputs.len() != originals.len() {
        return Err(mismatch("addressable operand count"));
    }
    for (array, layout) in originals.iter().zip(&quote.inputs) {
        if array.shape() != layout.shape()
            || crate::backend::nn::workspace::byte_view::Dtype::from_layout(layout.as_view())
                .is_none_or(|dtype| dtype.native() != array.dtype())
        {
            return Err(mismatch("addressable operand shape or dtype"));
        }
    }
    let parent = OriginalScopeObserver::require_current()?;
    access.validate_observer(&parent)?;
    if !access.metadata_custody()?.same_account(metadata){return Err(mismatch("addressable parent metadata account"));}
    let access = access.prepare_native_role(funding)?;
    let custody = Custody {
        raw: metadata.clone(),
        funding: funding.clone(),
    };
    let count = quote.outputs.len();
    let recipe = quote.numerical;
    let completion = NestedRootCompletion::prepare_metadata(
        count,
        recipe.completion.validation_roots,
        custody.raw.clone(),
        funding,
    )?;
    let collector = PreparedTokenChild::prepare_metadata(recipe.completion, custody.raw.clone(), &parent, funding)?;
    let mut clones = [
        PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        PreparedArrayClone::try_prepare_for_inspection()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
    ];
    // One actual enclosing-model completion precedes the child scope. All four
    // original operands then cross as settled frontiers, not foreign lazy DAGs.
    crate::backend::runtime::cache::complete_values(originals, stream)?;
    for array in originals {
        safemlx::OperationEvent::validate_traversal_leaf(array, &parent)?;
    }
    let inputs = [
        clones[0]
            .fill_for_inspection(originals[0])
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        clones[1]
            .fill_for_inspection(originals[1])
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        clones[2]
            .fill_for_inspection(originals[2])
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
        clones[3]
            .fill_for_inspection(originals[3])
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
    ];
    let plan = StreamCopyPlan::<Custody>::capture(stream).map_err(|_| mismatch("addressable stream inspection"))?;
    funding.reserve_metadata(StreamCopyPlan::<Custody>::constructor_storage_bytes()
        .map_err(|cause|Error::Neural(funding.metadata_source(cause)))?)
        .map_err(Error::WorkspacePlanning)?;
    let stream = plan.realize(custody.clone()).map_err(|_| mismatch("addressable stream realization"))?;
    let capacity = NativeRoleCapacity {
        graph: quote.capacity.graph,
        records: quote.capacity.records,
        backing: quote.capacity.backing,
    };
    let region = Region {
        inputs,
        completion: RefCell::new(completion),
        collector,
        quote,
        access,
        constructors: RefCell::new(Some(constructors)),
        reads,
        runtime: runtime.inspection_alias(),
        stream,
        custody: custody.clone(),
    };
    native_role::run_with_prepared_budget(
        region,
        capacity,
        Some(safemlx::PreparedPipelineCachePlan::new(recipe.kernels)),
        budget,
        &parent,
        &custody,
        funding,
        timeout,
        move |region, scope| {
            let recipe=region.quote.numerical;
            let observer = scope.observer();
            let collector = region.collector.enter(observer)?;
            for input in &region.inputs {
                safemlx::OperationEvent::validate_traversal_leaf(input, observer)?;
            }
            let mut graph =
                safemlx::OperationEvent::prepare_resident_graph(recipe.completion.graph, observer)?;
            let traversal = recipe.completion.nested_traversal().ok_or_else(identity)?;
            graph.configure_nested_completions(&traversal, recipe.completion.nested_completions)?;
            let access = region.access.bind(scope)?;
            let constructors = region
                .constructors
                .try_borrow_mut()
                .map_err(|_| identity())?
                .take()
                .ok_or_else(identity)?;
            let plan = region
                .quote
                .residency
                .clone_for_invocation(&region.custody.funding)?;
            let factory = plan.prepare_factory_shared(
                access,
                constructors,
                region.reads.clone(),
                &region.runtime,
                observer,
                region.stream.as_stream(),
            )?;
            let output = match run(factory) {
                Ok(output) => output,
                Err(cause) => {
                    drop(graph);
                    // Failed callbacks use the existing collector Drop path:
                    // restore the parent and retain every child root in Region.
                    // A secondary success-only scope check must not replace the
                    // original callback failure. Recovery still settles or
                    // quarantines the actual child before releasing authority.
                    drop(collector);
                    return Ok(Err(cause));
                }
            };
            let actual = 1 + usize::from(output.post_reduce().is_some());
            if actual != region.quote.outputs.len() {
                return Err(identity());
            }
            for (value, layout) in std::iter::once(output.reducible())
                .chain(output.post_reduce())
                .zip(&region.quote.outputs)
            {
                if value.as_array().shape() != layout.shape()
                    || crate::backend::nn::workspace::byte_view::Dtype::from_layout(
                        layout.as_view(),
                    )
                    .is_none_or(|dtype| dtype.native() != value.as_array().dtype())
                {
                    return Err(identity());
                }
            }
            region
                .completion
                .try_borrow_mut()
                .map_err(|_| identity())?
                .complete(
                    |visit| {
                        visit(output.reducible());
                        if let Some(bias) = output.post_reduce() {
                            visit(bias);
                        }
                        Ok(())
                    },
                    observer,
                    region.stream.as_stream(),
                )?;
            for value in std::iter::once(output.reducible()).chain(output.post_reduce()) {
                safemlx::OperationEvent::validate_traversal_leaf(value.as_array(), observer)?;
            }
            drop(graph);
            collector.finish()?;
            Ok(Ok(output))
        },
    )
    .map_err(|cause| Error::with_original_control_source(cause, false))?
}

mod provider;
pub(crate) use provider::{AddressableRequestOwner,AddressableExecutionRow};

pub(crate) mod request_sources;

mod speculative;
pub(crate) use speculative::{SpeculativeAddressableSources,SpeculativeAddressableSpan};

use eredu_nn::workspace::WorkspaceMetadataAllocation;
