use super::*;
use std::{cell::OnceCell, rc::Rc};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static NATIVE_COLLECTIVE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static CONTRACTED_COLLECTIVE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static ORIGINAL_MODEL_COLLECTIVE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_native_collective_submissions() {
    NATIVE_COLLECTIVE_SUBMISSIONS.with(|count| count.set(0));
    CONTRACTED_COLLECTIVE_SUBMISSIONS.with(|count| count.set(0));
    ORIGINAL_MODEL_COLLECTIVE_SUBMISSIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn native_collective_submissions() -> usize {
    NATIVE_COLLECTIVE_SUBMISSIONS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn contracted_collective_submissions() -> usize {
    CONTRACTED_COLLECTIVE_SUBMISSIONS.with(Cell::get)
}

// Only the exact retained model Sum/Gather producer records this category.
// Scheduler/control-world word exchanges cannot satisfy model-work evidence.
#[cfg(test)]
pub(crate) fn original_model_collective_submissions()->usize {
    ORIGINAL_MODEL_COLLECTIVE_SUBMISSIONS.with(Cell::get)
}
#[cfg(test)]
fn record_original_model_collective_submission(group:&Group) {
    record_native_collective_submission(group);
    ORIGINAL_MODEL_COLLECTIVE_SUBMISSIONS.with(|count|count.set(count.get()+1));
}

pub(super) fn record_native_collective_submission(_group: &Group) {
    #[cfg(test)]
    {
        NATIVE_COLLECTIVE_SUBMISSIONS.with(|count| count.set(count.get() + 1));
        if _group.contract.is_some() {
            CONTRACTED_COLLECTIVE_SUBMISSIONS.with(|count| count.set(count.get() + 1));
        }
    }
}

// Ordinary communicator setup retains the actual selected stream together with
// its existing scheduler/Event initialization observation. This is not request
// quota or numerical authority; native constructors still revalidate the worker.
struct PreparedTransportStream {
    stream: Stream,
    _runtime: safemlx::PrefillRootsRuntime,
}

/// A native MLX group or a backend-owned logical subgroup of one native world.
#[derive(Clone)]
pub struct Group {
    pub(super) native: native::Group,
    retained_buffer: Option<native::RetainedGroupBuffer>,
    transport_stream: Rc<OnceCell<PreparedTransportStream>>,
    pub(super) logical: Option<LogicalSubgroup>,
    pub(super) contract: Option<ManifestGroupContract>,
    pub(super) completion: Option<CommunicationCompletionPolicy>,
    source: Option<eredu_runtime::RetainedCommunicationSource>,
    original_parallel: Option<super::super::topology::original_source::parallel::OriginalParallelBinding>,
    original_control: Option<super::super::topology::original_source::control::OriginalControlBinding>,
    original_control_request: Option<super::super::topology::original_source::control::OriginalParallelControlProjection>,
}

#[derive(Debug, Clone)]
pub(super) struct ManifestGroupContract {
    pub(super) id: CollectiveGroupId,
    pub(super) requirements: CommunicationGroupRequirements,
}

#[derive(Debug, Clone)]
pub(super) struct LogicalSubgroup {
    pub(super) global_ranks: Vec<usize>,
    pub(super) rank: usize,
    pub(super) routes: Option<Vec<LogicalRoute>>,
    pub(super) world_collective_wave: bool,
}

#[derive(Debug, Clone)]
pub(super) struct LogicalRoute {
    pub(super) source_rank: usize,
    pub(super) exchanges: Vec<Option<usize>>,
}

impl Group {
    /// Wraps a control-plane native group without a manifest contract.
    pub fn uncontracted(group: &native::Group) -> Self {
        Self {
            native: group.clone(),
            retained_buffer: group.persistent_storage().ok().and_then(|source|source.retain_buffer().ok()),
            transport_stream: Rc::new(OnceCell::new()),
            logical: None,
            contract: None,
            completion: None,
            source: None,
            original_parallel: None,
            original_control: None,
            original_control_request: None,
        }
    }

    pub(crate) fn retained_buffer(&self)->Option<&native::RetainedGroupBuffer>{self.retained_buffer.as_ref()}

    /// Cold payload inventory of this actual idle group. Runtime invocation
    /// bindings may retain model/output roots, so a buffer-only census cannot
    /// certify them. Logical membership, manifests, retained source descriptors
    /// and the selected transport stream are control metadata; native Ring
    /// payload is represented by the independently retained setup buffer.
    pub(crate) fn collect_idle_retained_storage(
        &self,
        storage:&mut crate::backend::runtime::residency::storage::RetainedStorage,
    )->std::result::Result<(),crate::backend::runtime::residency::manager::ResidencyError>{
        // Exhaustive field classification makes a new group owner require an
        // explicit decision before it can release loading authority.
        let Self{native:_,retained_buffer,transport_stream:_,logical:_,contract:_,
            completion:_,source:_,original_parallel,original_control,original_control_request}=self;
        if original_parallel.is_some() || original_control.is_some() || original_control_request.is_some(){
            storage.mark_incomplete();
        }
        storage.include_group_buffer(retained_buffer.as_ref())
    }

    pub(crate) fn with_retained_source(mut self, source: eredu_runtime::RetainedCommunicationSource) -> Self {
        self.source = Some(source);
        self
    }
    pub(crate) fn retained_source(&self) -> Option<&eredu_runtime::RetainedCommunicationSource> {
        self.source.as_ref()
    }
    pub(crate) fn matches_retained_world(&self, source: &eredu_runtime::RetainedCommunicationSource) -> bool {
        self.source.as_ref().is_some_and(|actual| actual.same_source(source))
            && self.logical.is_none() && self.contract.is_none()
            && self.completion == source.manifest().completion_policy()
    }
    /// Borrow-only comparison against the actual selected group declaration.
    /// Native operation availability and finite work admission remain separate.
    pub(crate) fn matches_retained_group(&self, source: &eredu_runtime::RetainedCommunicationSource,
        descriptor: &CommunicationGroupDescriptor, world_wave: bool) -> bool {
        self.source.as_ref().is_some_and(|actual| actual.same_source(source))
            && self.completion == source.manifest().completion_policy()
            && self.contract.as_ref().is_some_and(|contract|
                contract.id == descriptor.id() && &contract.requirements == descriptor.requirements())
            && match &self.logical {
                Some(logical) => logical.global_ranks.as_slice() == descriptor.members()
                    && Some(logical.rank) == descriptor.local_index()
                    && logical.world_collective_wave == world_wave && logical.routes.is_none(),
                None => descriptor.members().len() == source.manifest().world_size()
                    && descriptor.members().iter().copied().eq(0..source.manifest().world_size())
                    && descriptor.local_index() == Some(source.manifest().rank()),
            }
    }

    pub(crate) fn matches_retained_route(&self, source: &eredu_runtime::RetainedCommunicationSource,
        descriptor: &eredu_runtime::CommunicationRouteDescriptor, world_wave: bool) -> bool {
        self.source.as_ref().is_some_and(|actual| actual.same_source(source))
            && self.completion == source.manifest().completion_policy()
            && self.contract.is_none()
            && self.logical.as_ref().is_some_and(|logical|
                logical.global_ranks.as_slice() == [descriptor.source(), descriptor.destination()]
                    && logical.rank == usize::from(source.manifest().rank() == descriptor.destination())
                    && logical.world_collective_wave == world_wave && logical.routes.is_none())
    }

    pub(crate) fn shares_native_world(&self, other: &Self) -> bool {
        self.native.shares_native_handle(&other.native)
    }

    /// Keeps the communicator-selected stream alive through every group clone,
    /// including groups retained by pending completion and recovery owners.
    pub(super) fn communication_stream<'a>(&'a self, compute: &'a Stream) -> Result<&'a Stream> {
        // Singleton collectives are identities and need no transport device.
        // Avoid initializing the native fallback's unrelated default device.
        if self.native.size() == 1 {
            return Ok(compute);
        }
        let preferred = self.initialize_transport_stream()?;
        // Preserve caller ordering and device affinity when the communicator
        // supports that device class (including a selected CUDA device).
        if preferred.get_device()?.get_type()? == compute.get_device()?.get_type()? {
            Ok(compute)
        } else {
            Ok(preferred)
        }
    }

    /// Initial setup and ordinary lazy use share this one native stream owner.
    /// A later original source may only borrow the completed setup field.
    pub(crate) fn initialize_transport_stream(&self) -> Result<&Stream> {
        if self.transport_stream.get().is_none() {
            let stream = self.native.communication_stream()?;
            // CPU compute may be distinct from the communicator's CPU stream.
            // Ordinary collectives can use the former, so they do not implicitly
            // initialize the latter. Prepare exactly the retained transport at
            // ordinary setup, before original construction can borrow it.
            let runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream)
                .map_err(Exception::from_source)?;
            let _ = self.transport_stream.set(PreparedTransportStream { stream, _runtime: runtime });
        }
        Ok(&self.transport_stream.get().expect("transport stream initialized").stream)
    }

    pub(crate) fn retained_transport_stream(&self) -> Option<&Stream> {
        self.transport_stream.get().map(|prepared| &prepared.stream)
    }

    pub(crate) fn with_completion_policy(
        mut self,
        completion: CommunicationCompletionPolicy,
    ) -> Self {
        self.completion = Some(completion);
        self
    }

    /// Terminal marking is a native incarnation operation, not a completion or
    /// neutral session policy decision. No production failure caller is added.
    pub(crate) fn mark_terminal_submission(&self) {
        self.native.mark_terminal_submission();
    }

    pub(super) fn ensure_available(&self) -> Result<()> {
        self.native
            .check_submission_available()
            .map_err(Exception::from_source)?;
        crate::backend::runtime::distributed::completion::ensure_group_available(self)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Attaches one exact opaque manifest identity and operation contract.
    pub(crate) fn with_manifest_contract(
        mut self,
        descriptor: &CommunicationGroupDescriptor,
        completion: CommunicationCompletionPolicy,
    ) -> Result<Self> {
        if self.size() != descriptor.members().len()
            || self.rank() != descriptor.local_index().unwrap_or(usize::MAX)
        {
            return Err(Exception::custom(format!(
                "opaque group {} native rank geometry differs from its manifest",
                descriptor.id().value()
            )));
        }
        self.contract = Some(ManifestGroupContract {
            id: descriptor.id(),
            requirements: descriptor.requirements().clone(),
        });
        self.completion = Some(completion);
        Ok(self)
    }

    /// Acquires the global runtime for one manifest-selected synchronous setup
    /// phase without allowing lock contention to outlive its policy.
    pub(crate) fn begin_bounded_setup(&self) -> Result<Option<safemlx::RuntimeCallGuard>> {
        self.ensure_available()?;
        self.completion
            .map(|policy| safemlx::RuntimeCallDeadline::new(policy.timeout())?.enter())
            .transpose()
    }

    pub(crate) fn completion_policy(&self) -> Option<CommunicationCompletionPolicy> {
        self.completion
    }

    #[cfg(test)]
    pub(crate) fn opaque_id(&self) -> Option<CollectiveGroupId> {
        self.contract.as_ref().map(|contract| contract.id)
    }

    pub(crate) const fn native_group(&self) -> &native::Group {
        &self.native
    }

    /// Returns this process's rank in this native or logical group.
    pub fn rank(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native.rank(), |logical| logical.rank)
    }

    /// Returns this native or logical group's size.
    pub fn size(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native.size(), |logical| logical.global_ranks.len())
    }

    /// Returns whether this is a backend-routed logical subgroup.
    pub fn is_logical(&self) -> bool {
        self.logical.is_some()
    }

    /// Attempts a backend-native split.
    pub fn split(&self, color: i32, key: Option<i32>) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "backend-native splitting of a logical subgroup is unsupported",
            ));
        }
        let native=self.native.split(color,key)?;
        let retained_buffer=native.persistent_storage().ok().and_then(|source|source.retain_buffer().ok());
        Ok(Self {
            native,retained_buffer,
            transport_stream: Rc::new(OnceCell::new()),
            logical: None,
            contract: None,
            completion: self.completion,
            source: self.source.clone(),
            original_parallel: self.original_parallel.clone(),
            original_control: self.original_control.clone(),
            original_control_request: self.original_control_request.clone(),
        })
    }

    /// Creates a logical subgroup over the same native world group.
    pub fn logical_subgroup(&self, global_ranks: &[usize]) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "logical subgroups must be derived from a native world group",
            ));
        }
        if global_ranks.is_empty() {
            return Err(Exception::custom("logical subgroup cannot be empty"));
        }
        let native_size = self.native.size();
        let mut seen = vec![false; native_size];
        for &rank in global_ranks {
            if rank >= native_size {
                return Err(Exception::custom(format!(
                    "logical subgroup rank {rank} is outside native world size {native_size}"
                )));
            }
            if std::mem::replace(&mut seen[rank], true) {
                return Err(Exception::custom(format!(
                    "logical subgroup repeats global rank {rank}"
                )));
            }
        }
        let native_rank = self.native.rank();
        let rank = global_ranks
            .iter()
            .position(|rank| *rank == native_rank)
            .ok_or_else(|| {
                Exception::custom(format!(
                    "native rank {native_rank} is not a member of logical subgroup {global_ranks:?}"
                ))
            })?;
        Ok(Self {
            native: self.native.clone(),
            retained_buffer: self.retained_buffer.clone(),
            transport_stream: Rc::clone(&self.transport_stream),
            logical: Some(LogicalSubgroup {
                global_ranks: global_ranks.to_vec(),
                rank,
                routes: None,
                world_collective_wave: false,
            }),
            contract: None,
            completion: self.completion,
            source: self.source.clone(),
            original_parallel: self.original_parallel.clone(),
            original_control: self.original_control.clone(),
            original_control_request: self.original_control_request.clone(),
        })
    }

    /// Creates a logical subgroup with topology-planned native-world routes.
    pub fn logical_subgroup_with_routes(
        &self,
        global_ranks: &[usize],
        routes: Vec<(usize, Vec<Option<usize>>)>,
    ) -> Result<Self> {
        let mut group = self.logical_subgroup(global_ranks)?;
        let native_rank = group.native.rank();
        let native_size = group.native.size();
        let mut seen = vec![false; global_ranks.len()];
        let routes = routes
            .into_iter()
            .map(|(source_rank, exchanges)| {
                if source_rank >= global_ranks.len()
                    || std::mem::replace(&mut seen[source_rank], true)
                {
                    return Err(Exception::custom(format!(
                        "logical route source rank {source_rank} is missing or repeated for subgroup size {}",
                        global_ranks.len()
                    )));
                }
                for peer in exchanges.iter().flatten() {
                    if *peer >= native_size
                        || !((native_rank + 1) % native_size == *peer
                            || (*peer + 1) % native_size == native_rank)
                    {
                        return Err(Exception::custom(format!(
                            "logical route from native rank {native_rank} uses non-neighbor peer {peer}"
                        )));
                    }
                }
                Ok(LogicalRoute {
                    source_rank,
                    exchanges,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if seen.iter().any(|present| !present) {
            return Err(Exception::custom(
                "logical routes do not cover every subgroup source rank",
            ));
        }
        group.logical.as_mut().expect("logical subgroup").routes = Some(routes);
        Ok(group)
    }

    pub(crate) fn with_world_collective_wave(mut self, proven: bool) -> Self {
        if let Some(logical) = &mut self.logical {
            logical.world_collective_wave = proven;
        }
        self
    }
}

fn tensor_dtype(dtype: Dtype) -> TensorDtype {
    match dtype {
        Dtype::Bool => TensorDtype::Bool,
        Dtype::Uint8 => TensorDtype::U8,
        Dtype::Uint16 => TensorDtype::U16,
        Dtype::Uint32 => TensorDtype::U32,
        Dtype::Uint64 => TensorDtype::U64,
        Dtype::Int8 => TensorDtype::I8,
        Dtype::Int16 => TensorDtype::I16,
        Dtype::Int32 => TensorDtype::I32,
        Dtype::Int64 => TensorDtype::I64,
        Dtype::Float16 => TensorDtype::F16,
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float64 => TensorDtype::F64,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        Dtype::Complex64 => TensorDtype::Complex64,
    }
}

impl Group {
    fn requirement(
        &self,
        operation: CommunicationOperation,
    ) -> Result<Option<&CommunicationOperationRequirement>> {
        self.ensure_available()?;
        let Some(contract) = &self.contract else {
            return Ok(None);
        };
        let requirement = contract
            .requirements
            .operations()
            .iter()
            .find(|requirement| requirement.operation() == operation)
            .ok_or_else(|| {
                Exception::custom(format!(
                    "opaque group {} does not select operation {operation:?}",
                    contract.id.value()
                ))
            })?;
        if !requirement.exact_completion() {
            return Err(Exception::custom(format!(
                "opaque group {} selects inexact operation {operation:?}",
                contract.id.value()
            )));
        }
        Ok(Some(requirement))
    }

    pub(crate) fn validate_tensor(
        &self,
        operation: CommunicationOperation,
        value: &Array,
        completed: bool,
    ) -> Result<()> {
        let Some(requirement) = self.requirement(operation)? else {
            return Ok(());
        };
        requirement.limits().ok_or_else(|| {
            Exception::custom(format!("operation {operation:?} has no tensor limits"))
        })?;
        let dtype = tensor_dtype(value.dtype());
        requirement.validate_tensor_metadata(&dtype,value.ndim(),Some(value.size()),completed)
            .map_err(|error| match error {
                eredu_runtime::CommunicationTensorContractError::Dtype=>Exception::custom(format!(
                    "opaque group contract for {operation:?} does not admit dtype {dtype:?}")),
                eredu_runtime::CommunicationTensorContractError::MissingLimits=>Exception::custom(format!(
                    "operation {operation:?} has no tensor limits")),
                eredu_runtime::CommunicationTensorContractError::Limits=>Exception::custom(format!(
                    "opaque group contract for {operation:?} rejects tensor shape {:?}",value.shape())),
            })
    }

    pub(super) fn validate_payload_free(&self, operation: CommunicationOperation) -> Result<()> {
        self.requirement(operation).map(|_| ())
    }

    pub(crate) fn validate_expected_output(
        &self,
        operation: CommunicationOperation,
        dtype: Dtype,
        rank: usize,
        elements: usize,
    ) -> Result<()> {
        let Some(requirement) = self.requirement(operation)? else {
            return Ok(());
        };
        let limits = requirement.limits().ok_or_else(|| {
            Exception::custom(format!("operation {operation:?} has no tensor limits"))
        })?;
        let dtype = tensor_dtype(dtype);
        if !requirement.dtypes().contains(&dtype)
            || limits.max_tensors() < 1
            || rank > limits.max_tensor_rank()
            || elements > limits.max_output_tensor_elements()
        {
            return Err(Exception::custom(format!(
                "opaque group contract for {operation:?} rejects expected output rank {rank}, elements {elements}, dtype {dtype:?}"
            )));
        }
        Ok(())
    }

    pub(super) fn validate_peer_counts(
        &self,
        operation: CommunicationOperation,
        send_counts: &[usize],
        receive_counts: &[usize],
    ) -> Result<()> {
        let Some(requirement) = self.requirement(operation)? else {
            return Ok(());
        };
        let maximum = requirement
            .limits()
            .and_then(|limits| limits.max_count_per_peer())
            .ok_or_else(|| {
                Exception::custom("variable exchange contract has no peer-count limit")
            })?;
        if send_counts.len() != self.size()
            || receive_counts.len() != self.size()
            || send_counts
                .iter()
                .chain(receive_counts)
                .any(|count| *count > maximum)
        {
            return Err(Exception::custom(
                "opaque group variable-exchange peer counts exceed the selected contract",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Group")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .field("logical", &self.is_logical())
            .field(
                "opaque_id",
                &self.contract.as_ref().map(|contract| contract.id),
            )
            .finish()
    }
}

#[cfg(test)]
mod terminal_tests;

mod retention_copy;

impl Group {
    pub(crate) fn original_control_request(&self)->Option<&super::super::topology::original_source::control::OriginalParallelControlProjection> {
        self.original_control_request.as_ref()
    }
    pub(crate) fn validate_control_model_source(&self,source:&super::super::topology::original_source::parallel::OriginalParallelSource)
        ->std::result::Result<(),crate::backend::error::Error> {
        match &self.original_parallel {
            Some(binding)=>binding.validate_control_source(self,source),
            None=>Err(crate::backend::error::Error::Neural(source.neural_error(
                crate::backend::error::Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)))),
        }
    }
    /// Only the paid model-context loan attaches its authenticated request view.
    pub(crate) fn bind_original_control_request(&mut self,
        projection:super::super::topology::original_source::control::OriginalParallelControlProjection) {
        self.original_control_request=Some(projection);
    }
    /// The prepared publication group is an alias of the exact Q invocation.
    pub(crate) fn with_prepared_publication_group<T,E,F>(&self,ordinary:&Group,
        funding:&eredu_nn::workspace::HostMetadataFunding,stream:&Stream,run:F)
        ->std::result::Result<std::result::Result<T,E>,crate::backend::error::Error>
    where F:FnOnce(Option<&Group>)->std::result::Result<T,E> {
        match &self.original_parallel {
            Some(binding)=>binding.with_publication_group(ordinary,self,funding,stream,run),
            None=>Ok(run(None)),
        }
    }
    pub(crate) fn publish_original(&self,input:&Array,root:usize,stream:&Stream)
        ->Option<std::result::Result<(Array,super::super::completion::OriginalCommunicationCompletion),crate::backend::error::Error>> {
        self.original_parallel.as_ref().map(|binding|binding.publish(self,input,root,stream).map(|(value,completion)| {
            let (value,source,funding)=value.into_parts();
            // Completion owns the same accepted source/H, while Q retains the
            // invocation through final model completion or failed recovery.
            // The native Array retains its actual Graph/backing allocations.
            drop((source,funding));
            (value,completion)
        }))
    }
    pub(crate) fn original_parallel_binding(&self)
        ->Option<&super::super::topology::original_source::parallel::OriginalParallelBinding>{
        self.original_parallel.as_ref()
    }
    pub(crate) fn has_original_parallel(&self)->bool {self.original_parallel.is_some()}
    pub(crate) fn with_original_parallel(mut self,binding:super::super::topology::original_source::parallel::OriginalParallelBinding)->Self {
        self.original_parallel=Some(binding);self
    }
    pub(crate) fn model_funding(&self)->Option<&eredu_nn::workspace::HostMetadataFunding> {
        self.original_parallel.as_ref().map(|binding|binding.funding())
    }
    pub(crate) fn model_gather_error(&self,cause:eredu_nn::ParallelGatherError)->eredu_nn::Error {
        match &self.original_parallel {
            Some(binding)=>binding.error(super::super::topology::original_source::Cause::Gather(cause)),
            None=>eredu_nn::Error::backend_retained_source(cause),
        }
    }
    pub(crate) fn model_embedding_error(&self,cause:eredu_nn::EmbeddingValidationError)->eredu_nn::Error {
        match &self.original_parallel {
            Some(binding)=>binding.error(super::super::topology::original_source::Cause::Embedding(cause)),
            None=>eredu_nn::Error::backend(cause),
        }
    }
    pub(crate) fn model_source_missing(&self)->eredu_nn::Error {
        match &self.original_parallel {
            Some(binding)=>binding.error(super::super::topology::original_source::Cause::Resource),
            None=>eredu_nn::Error::backend("embedding has no vocabulary ownership"),
        }
    }
    pub(crate) fn model_vocabulary_error(&self,cause:eredu_nn::VocabularyRangeError)->eredu_nn::Error {
        match &self.original_parallel {
            Some(binding)=>binding.error(super::super::topology::original_source::Cause::Vocabulary(cause)),
            None=>eredu_nn::Error::backend_retained_source(cause),
        }
    }
    pub(crate) fn model_native_error(&self,cause:Exception)->eredu_nn::Error {
        match &self.original_parallel {
            Some(binding)=>binding.error(super::super::topology::original_source::Cause::Native(cause)),
            None=>eredu_nn::Error::backend_retained_source(cause),
        }
    }
    pub(crate) fn gather_model_first(&self,input:&Array,stream:&Stream,axis:usize,widths:&[usize])->std::result::Result<Array,eredu_nn::Error> {
        match &self.original_parallel {
            Some(binding)=>{
                let output=binding.gather(self,input,stream,axis,widths)?;
                #[cfg(test)]
                record_original_model_collective_submission(self);
                Ok(output)
            },
            None=>super::all_gather_unchecked(input,self,stream).map_err(eredu_nn::Error::backend_retained_source),
        }
    }
    /// Ordinary model Sum or the explicitly bound occurrence in this exact
    /// parallel context. This does not create an intermediate completion.
    pub(crate) fn sum_model(&self,input:&Array,stream:&Stream)->std::result::Result<Array,eredu_nn::Error> {
        match &self.original_parallel {
            Some(binding)=>{
                let output=binding.sum(self,input,stream)?;
                #[cfg(test)]
                record_original_model_collective_submission(self);
                Ok(output)
            },
            None=>super::all_sum(input,self,stream).map_err(eredu_nn::Error::backend_retained_source),
        }
    }
}

impl Group {
    pub(crate) fn has_original_control(&self)->bool {self.original_control.is_some()}
    pub(crate) fn original_control(&self)->Option<&super::super::topology::original_source::control::OriginalControlBinding> {
        self.original_control.as_ref()
    }
    pub(crate) fn with_original_control(mut self,binding:super::super::topology::original_source::control::OriginalControlBinding)->Self {
        self.original_control=Some(binding);self
    }
}


impl Group {
    /// Authenticate the ordinary opaque handle against this exact model source.
    /// Execution still consumes the retained occurrence and validates its scope.
    pub(crate) fn validate_model_collective_group(&self, ordinary:&Group)
        ->std::result::Result<(),eredu_nn::Error>{
        let fail=||self.model_source_missing();
        let source=self.retained_source().ok_or_else(fail)?;
        let id=self.contract.as_ref().ok_or_else(fail)?.id;
        let descriptor=source.manifest().groups().iter().find(|entry|entry.id()==id).ok_or_else(fail)?;
        let wave=source.realization().group_world_wave(descriptor.creation_order()).ok_or_else(fail)?;
        if !self.has_original_parallel()
            || !self.native.shares_native_handle(&ordinary.native)
            || !self.matches_retained_group(source,descriptor,wave)
            || !ordinary.matches_retained_group(source,descriptor,wave) {
            return Err(fail());
        }
        Ok(())
    }
}
