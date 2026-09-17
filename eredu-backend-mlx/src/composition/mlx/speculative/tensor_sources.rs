//! Exact immutable tensor evidence lent lexically to one model invocation.
use super::{Error, OriginalSpeculativeNumericalSources, RegisteredTensorSource};
use crate::backend::{OriginalCopyEnvironment, runtime::cache::state::CompletedResidentSource};
use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
use eredu_runtime::working_memory::WorkingMemoryError;

/// Borrows the actual completed inventory. Registered copies keep their existing
/// canonical source pins; unknown evidence cannot stand in for either source.
pub(in crate::composition::mlx) fn completed_tensor_source(
    evidence: &PreparedEmbeddedEvidence,
) -> Option<&CompletedResidentSource> {
    evidence.get::<CompletedResidentSource>().or_else(||evidence.get::<super::CompletedTensorSource>().map(|source|&source.source)).or_else(|| evidence.get::<super::ExternalCompletedSource>().map(|source| &source.source))
}
pub(super) fn validate_tensor_sources(
    evidence: &[&PreparedEmbeddedEvidence],
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
) -> Result<(), Error> {
    validate_tensor_inventory(evidence,sources,environment,None)
}
/// A lexical inventory names completed producers, before an operation selects
/// its consuming side. Authenticate each witness against the two retained
/// environments; the later input binder still requires the selected placement.
pub(super) fn validate_external_tensor_sources(
    evidence:&[&PreparedEmbeddedEvidence],context:super::SpeculativeExecutionStreams<'_>,
)->Result<(),Error>{
    use eredu_core::speculative::SamplingPlacement;
    let invalid=||Error::PrefillControl(WorkingMemoryError::IdentityMismatch);
    let (sources,target)=context.original_numerical_for(SamplingPlacement::Target).ok_or_else(invalid)?;
    let (draft_sources,draft)=context.original_numerical_for(SamplingPlacement::Draft).ok_or_else(invalid)?;
    if context.original_external().is_none() || !std::ptr::eq(sources,draft_sources)
        || !target.pool().same_domain(draft.pool()) {return Err(invalid());}
    validate_tensor_inventory(evidence,sources,target,Some(draft))
}
fn validate_tensor_inventory(
    evidence:&[&PreparedEmbeddedEvidence],sources:&OriginalSpeculativeNumericalSources,
    environment:&OriginalCopyEnvironment<'_>,draft:Option<&OriginalCopyEnvironment<'_>>,
)->Result<(),Error>{
    let parts=[std::mem::size_of::<RegisteredTensorSources<'_>>(),
        std::mem::size_of::<std::slice::Iter<'_,&PreparedEmbeddedEvidence>>(),
        std::mem::size_of::<Result<(),Error>>(),std::mem::size_of::<bool>(),
        std::mem::size_of::<(&[&PreparedEmbeddedEvidence],super::SpeculativeExecutionStreams<'_>)>(),
        std::mem::size_of::<(&[&PreparedEmbeddedEvidence],&OriginalSpeculativeNumericalSources,
            &OriginalCopyEnvironment<'_>,Option<&OriginalCopyEnvironment<'_>>)>(),
        std::mem::size_of::<[(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>);2]>(),
        std::mem::size_of::<InputSource<'_>>(),
        std::mem::size_of::<(InputSource<'_>,&OriginalSpeculativeNumericalSources,
            &OriginalCopyEnvironment<'_>,Option<&OriginalCopyEnvironment<'_>>)>(),
        std::mem::size_of::<Option<&OriginalCopyEnvironment<'_>>>(),
        std::mem::size_of::<Result<bool,Error>>(),std::mem::size_of::<Result<(),Error>>()];
    sources.metadata_funding().reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    sources.validate_environment(environment)?;
    if let Some(draft)=draft {sources.validate_environment(draft)?;}
    for value in evidence {
        let mut known=false;
        if let Some(completed)=completed_tensor_source(value) {
            validate_inventory_source(InputSource::Completed(completed),sources,environment,draft)?;known=true;
        }
        for registered in registered_tensor_sources(value){
            validate_inventory_source(InputSource::Registered(registered),sources,environment,draft)?;known=true;
        }
        if !known{return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));}
    }
    Ok(())
}

fn validate_inventory_source(source:InputSource<'_>,sources:&OriginalSpeculativeNumericalSources,
    target:&OriginalCopyEnvironment<'_>,draft:Option<&OriginalCopyEnvironment<'_>>,
)->Result<(),Error>{
    let funding=sources.metadata_funding();
    // Ordinary Embedded inventory remains exact to Target. External inventory
    // may retain an already completed copy from Draft, without relabeling it.
    let origin=match draft {
        Some(draft) if !source.matches(target.stream(),funding)?=>{
            if !source.matches(draft.stream(),funding)? {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            draft
        },
        _=>target,
    };
    source.validate(sources,origin.stream(),funding)
}

pub(in crate::composition::mlx) fn registered_tensor_source(evidence:&PreparedEmbeddedEvidence)
    ->Option<&RegisteredTensorSource>{evidence.get::<RegisteredTensorSource>()}
/// Finite current-root witnesses; predecessor history grants no extra roots.
pub(in crate::composition::mlx) struct RegisteredTensorSources<'a> {
    direct:Option<&'a RegisteredTensorSource>,
    shared:std::slice::Iter<'a,RegisteredTensorSource>,
}
impl<'a> Iterator for RegisteredTensorSources<'a>{
    type Item=&'a RegisteredTensorSource;
    fn next(&mut self)->Option<Self::Item>{self.direct.take().or_else(||self.shared.next())}
}
pub(in crate::composition::mlx) fn registered_tensor_sources(evidence:&PreparedEmbeddedEvidence)
    ->RegisteredTensorSources<'_>{
    RegisteredTensorSources{direct:registered_tensor_source(evidence),shared:evidence.get::<super::ExternalCompletedSource>()
        .map_or(&[][..],|source|source.registered.as_slice()).iter()}
}
pub(super) fn registered_source_for_array<'a>(evidence:&'a PreparedEmbeddedEvidence,array:&safemlx::Array,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding)->Result<Option<&'a RegisteredTensorSource>,Error>{
    let parts=[std::mem::size_of::<RegisteredTensorSources<'_>>(),
        std::mem::size_of::<Option<&RegisteredTensorSource>>(),
        std::mem::size_of::<Result<Option<&RegisteredTensorSource>,Error>>(),
        std::mem::size_of::<(&PreparedEmbeddedEvidence,&safemlx::Array,&eredu_nn::workspace::WorkspaceMetadataFunding)>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let mut found=None;
    for source in registered_tensor_sources(evidence){
        if source.matches_array(array,funding)?{found=Some(source);}
    }
    Ok(found)
}
/// Join each registered packet's exact backing with the actual native input
/// projection. The existing binder still acquires its canonical registered pin.
pub(in crate::composition::mlx) fn validate_registered_tensor_inputs(
    context:super::SpeculativeExecutionStreams<'_>,native:&crate::backend::nn::workspace::ProjectedNativeStorage,
)->Result<(),Error>{
    let (sources,_)=context.original_numerical()
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    validate_registered_inputs(context,native,None,sources.metadata_funding())
}
/// The selected model still binds every actual input account. A retained
/// registered witness may name either authenticated completed input side.
pub(in crate::composition::mlx) fn validate_registered_tensor_inputs_at(
    context:super::SpeculativeExecutionStreams<'_>,native:&crate::backend::nn::workspace::ProjectedNativeStorage,
    placement:eredu_core::speculative::SamplingPlacement,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<(),Error>{
    validate_registered_inputs(context,native,Some(placement),funding)
}
fn validate_registered_inputs(
    context:super::SpeculativeExecutionStreams<'_>,native:&crate::backend::nn::workspace::ProjectedNativeStorage,
    selected:Option<eredu_core::speculative::SamplingPlacement>,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<(),Error>{
    let (sources,environment)=context.original_numerical_for(selected.unwrap_or(eredu_core::speculative::SamplingPlacement::Target))
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
    let parts=[std::mem::size_of::<RegisteredTensorSources<'_>>(),
        std::mem::size_of::<std::slice::Iter<'_,&PreparedEmbeddedEvidence>>(),
        std::mem::size_of::<Result<(),Error>>(),
        std::mem::size_of::<(super::SpeculativeExecutionStreams<'_>,&crate::backend::nn::workspace::ProjectedNativeStorage,
            Option<eredu_core::speculative::SamplingPlacement>,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<(super::SpeculativeExecutionStreams<'_>,&crate::backend::nn::workspace::ProjectedNativeStorage,
            Option<eredu_core::speculative::SamplingPlacement>,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<InputSource<'_>>(),std::mem::size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    for evidence in context.embedded_tensor_sources(){
        for source in registered_tensor_sources(evidence){
            if let Some(placement)=selected {
                input_environment(InputSource::Registered(source),context,placement,funding)?;
            } else {
                source.validate(sources.request(),environment.stream(),funding)?;
            }
            source.validate_selected_projection(native,funding)?;
        }
    }
    Ok(())
}

// The marker names completed work on the source stream. A same-device alias
// keeps that marker; it never grants completion on the consuming stream.
#[derive(Clone, Copy)]
enum InputSource<'a> {
    Completed(&'a CompletedResidentSource),
    Registered(&'a RegisteredTensorSource),
}
impl InputSource<'_> {
    fn matches(self, stream:&safemlx::Stream,
        funding:&eredu_nn::workspace::WorkspaceMetadataFunding)->Result<bool,Error>{
        match self {
            Self::Completed(source)=>Ok(source.matches_completed_stream(stream,funding)?),
            Self::Registered(source)=>source.matches_completed_stream(stream,funding),
        }
    }
    fn validate(self, sources:&OriginalSpeculativeNumericalSources,
        stream:&safemlx::Stream, funding:&eredu_nn::workspace::WorkspaceMetadataFunding)->Result<(),Error>{
        match self {
            Self::Completed(source)=>Ok(source.validate_request_source(sources.request(),stream,funding)?),
            Self::Registered(source)=>source.validate(sources.request(),stream,funding),
        }
    }
}
fn input_environment<'a>(source:InputSource<'_>, context:super::SpeculativeExecutionStreams<'a>,
    placement:eredu_core::speculative::SamplingPlacement,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<&'a OriginalCopyEnvironment<'a>,Error>{
    use eredu_core::speculative::{SamplingPlacement,SpeculativeExecutionTopology};
    let parts=[std::mem::size_of::<(InputSource<'_>,super::SpeculativeExecutionStreams<'_>,
            SamplingPlacement,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<Option<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>)>>(),
        std::mem::size_of::<[(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>);2]>(),
        std::mem::size_of::<SamplingPlacement>(),std::mem::size_of::<Result<bool,Error>>(),
        std::mem::size_of::<Result<(),Error>>(),
        std::mem::size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>(),
        // Both closed dispatch helpers borrow the actual source; no retained
        // projection or new completed marker is allocated here.
        std::mem::size_of::<(InputSource<'_>,&safemlx::Stream,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<(InputSource<'_>,&OriginalSpeculativeNumericalSources,&safemlx::Stream,
            &eredu_nn::workspace::WorkspaceMetadataFunding)>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let invalid=||Error::PrefillControl(WorkingMemoryError::IdentityMismatch);
    let (sources,destination)=context.original_numerical_for(placement).ok_or_else(invalid)?;
    sources.validate_environment(destination)?;
    if source.matches(destination.stream(),funding)? {
        source.validate(sources,destination.stream(),funding)?;
        return Ok(destination);
    }
    if context.original_external().is_none()
        || context.topology()!=SpeculativeExecutionTopology::SameDeviceSplit {
        return Err(invalid());
    }
    let opposite=match placement {
        SamplingPlacement::Target=>SamplingPlacement::Draft,
        SamplingPlacement::Draft=>SamplingPlacement::Target,
        _=>return Err(invalid()),
    };
    let (other,origin)=context.original_numerical_for(opposite).ok_or_else(invalid)?;
    if !std::ptr::eq(sources,other) || !destination.pool().same_domain(origin.pool())
        || !source.matches(origin.stream(),funding)? {return Err(invalid());}
    sources.validate_environment(origin)?;
    source.validate(sources,origin.stream(),funding)?;
    Ok(origin)
}
/// Descriptive completed origin loan for a source-preserving state projection.
pub(super) fn completed_input_environment<'a>(source:&CompletedResidentSource,
    context:super::SpeculativeExecutionStreams<'a>,placement:eredu_core::speculative::SamplingPlacement,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<&'a OriginalCopyEnvironment<'a>,Error>{
    let parts=[std::mem::size_of::<(&CompletedResidentSource,super::SpeculativeExecutionStreams<'_>,
        eredu_core::speculative::SamplingPlacement,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    input_environment(InputSource::Completed(source),context,placement,funding)
}
/// Authenticates one actual backing at its completed source placement. Only an
/// already bound external same-device assignment may borrow the opposite side.
/// The returned loan does not change evidence or authorize a new operation.
pub(super) fn input_array_environment<'a>(evidence:&PreparedEmbeddedEvidence,array:&safemlx::Array,
    context:super::SpeculativeExecutionStreams<'a>,placement:eredu_core::speculative::SamplingPlacement,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<&'a OriginalCopyEnvironment<'a>,Error>{
    let parts=[std::mem::size_of::<(&PreparedEmbeddedEvidence,&safemlx::Array,
            super::SpeculativeExecutionStreams<'_>,eredu_core::speculative::SamplingPlacement,
            &eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<Option<&RegisteredTensorSource>>(),std::mem::size_of::<InputSource<'_>>(),
        std::mem::size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>(),
        std::mem::size_of::<Result<(),Error>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let source=if let Some(source)=registered_source_for_array(evidence,array,funding)? {
        InputSource::Registered(source)
    } else {
        InputSource::Completed(completed_tensor_source(evidence)
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?)
    };
    let environment=input_environment(source,context,placement,funding)?;
    match source {
        InputSource::Registered(source)=>source.validate_array(array,funding)?,
        InputSource::Completed(source)=>{
            funding.reserve_metadata(CompletedResidentSource::array_source_control_bytes()
                .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
                .map_err(Error::WorkspacePlanning)?;
            source.array_source_account(array,funding)?;
        },
    }
    Ok(environment)
}

/// Validate all current evidence declarations without treating unrelated prior
/// roots as part of this operation's actual native input projection.
pub(in crate::composition::mlx) fn validate_input_evidence(evidence:&PreparedEmbeddedEvidence,
    context:super::SpeculativeExecutionStreams<'_>,placement:eredu_core::speculative::SamplingPlacement,
    funding:&eredu_nn::workspace::WorkspaceMetadataFunding,
)->Result<(),Error>{
    let parts=[std::mem::size_of::<(&PreparedEmbeddedEvidence,super::SpeculativeExecutionStreams<'_>,
            eredu_core::speculative::SamplingPlacement,&eredu_nn::workspace::WorkspaceMetadataFunding)>(),
        std::mem::size_of::<RegisteredTensorSources<'_>>(),std::mem::size_of::<InputSource<'_>>(),
        std::mem::size_of::<Option<&CompletedResidentSource>>(),std::mem::size_of::<bool>(),
        std::mem::size_of::<Result<&OriginalCopyEnvironment<'_>,Error>>(),
        std::mem::size_of::<Result<(),Error>>()];
    funding.reserve_metadata(parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
        .ok_or(Error::WorkspacePlanning(eredu_nn::workspace::WorkspaceMetadataFundingError::Overflow))?)
        .map_err(Error::WorkspacePlanning)?;
    let mut known=false;
    if let Some(source)=completed_tensor_source(evidence){
        input_environment(InputSource::Completed(source),context,placement,funding)?;
        known=true;
    }
    for source in registered_tensor_sources(evidence){
        input_environment(InputSource::Registered(source),context,placement,funding)?;
        known=true;
    }
    if !known{return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));}
    Ok(())
}
