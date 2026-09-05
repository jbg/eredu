use super::*;

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static NATIVE_COLLECTIVE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static CONTRACTED_COLLECTIVE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_native_collective_submissions() {
    NATIVE_COLLECTIVE_SUBMISSIONS.with(|count| count.set(0));
    CONTRACTED_COLLECTIVE_SUBMISSIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn native_collective_submissions() -> usize {
    NATIVE_COLLECTIVE_SUBMISSIONS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn contracted_collective_submissions() -> usize {
    CONTRACTED_COLLECTIVE_SUBMISSIONS.with(Cell::get)
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

/// A native MLX group or a backend-owned logical subgroup of one native world.
#[derive(Clone)]
pub struct Group {
    pub(super) native: native::Group,
    pub(super) logical: Option<LogicalSubgroup>,
    pub(super) contract: Option<ManifestGroupContract>,
    pub(super) completion: Option<CommunicationCompletionPolicy>,
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
            logical: None,
            contract: None,
            completion: None,
        }
    }

    pub(crate) fn shares_native_world(&self, other: &Self) -> bool {
        self.native.shares_native_handle(&other.native)
    }

    pub(crate) fn with_completion_policy(
        mut self,
        completion: CommunicationCompletionPolicy,
    ) -> Self {
        self.completion = Some(completion);
        self
    }

    pub(super) fn ensure_available(&self) -> Result<()> {
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
        Ok(Self {
            native: self.native.split(color, key)?,
            logical: None,
            contract: None,
            completion: self.completion,
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
            logical: Some(LogicalSubgroup {
                global_ranks: global_ranks.to_vec(),
                rank,
                routes: None,
                world_collective_wave: false,
            }),
            contract: None,
            completion: self.completion,
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
        let limits = requirement.limits().ok_or_else(|| {
            Exception::custom(format!("operation {operation:?} has no tensor limits"))
        })?;
        let dtype = tensor_dtype(value.dtype());
        if !requirement.dtypes().contains(&dtype) {
            return Err(Exception::custom(format!(
                "opaque group contract for {operation:?} does not admit dtype {dtype:?}"
            )));
        }
        let maximum = if completed {
            limits.max_output_tensor_elements()
        } else {
            limits.max_tensor_elements()
        };
        if limits.max_tensors() < 1
            || value.ndim() > limits.max_tensor_rank()
            || value.size() > maximum
        {
            return Err(Exception::custom(format!(
                "opaque group contract for {operation:?} rejects tensor shape {:?}",
                value.shape()
            )));
        }
        Ok(())
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
